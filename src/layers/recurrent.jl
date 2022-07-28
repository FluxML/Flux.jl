
gate(h, n) = (1:h) .+ h*(n-1)
gate(x::AbstractVector, h, n) = @view x[gate(h,n)]
gate(x::AbstractMatrix, h, n) = view(x, gate(h,n), :)

# AD-friendly helper for dividing monolithic RNN params into equally sized gates
multigate(x::AbstractArray, h, ::Val{N}) where N = ntuple(n -> gate(x,h,n), N)

function ChainRulesCore.rrule(::typeof(multigate), x::AbstractArray, h, c)
  function multigate_pullback(dy)
    dx = map!(zero, similar(x, float(eltype(x)), axes(x)), x)
    foreach(multigate(dx, h, c), dy) do dxᵢ, dyᵢ
      dyᵢ isa AbstractZero && return
      @. dxᵢ += dyᵢ
    end
    return (NoTangent(), dx, NoTangent(), NoTangent())
  end
  return multigate(x, h, c), multigate_pullback
end

# Type stable and AD-friendly helper for iterating over the last dimension of an array
function eachlastdim(A::AbstractArray{T,N}) where {T,N}
  inds_before = ntuple(_ -> :, N-1)
  return (view(A, inds_before..., i) for i in axes(A, N))
end

# adapted from https://github.com/JuliaDiff/ChainRules.jl/blob/f13e0a45d10bb13f48d6208e9c9d5b4a52b96732/src/rulesets/Base/indexing.jl#L77
function ∇eachlastdim(dys_raw, x::AbstractArray{T, N}) where {T, N}
  dys = unthunk(dys_raw)
  i1 = findfirst(dy -> dy isa AbstractArray, dys)
  if isnothing(i1)  # all slices are Zero!
      return fill!(similar(x, T, axes(x)), zero(T))
  end
  # The whole point of this gradient is that we can allocate one `dx` array:
  dx = similar(x, T, axes(x))::AbstractArray
  for i in axes(x, N)
      slice = selectdim(dx, N, i)
      if dys[i] isa AbstractZero
          fill!(slice, zero(eltype(slice)))
      else
          copyto!(slice, dys[i])
      end
  end
  return ProjectTo(x)(dx)
end

function ChainRulesCore.rrule(::typeof(eachlastdim), x::AbstractArray{T,N}) where {T,N}
  lastdims(dy) = (NoTangent(), ∇eachlastdim(unthunk(dy), x))
  collect(eachlastdim(x)), lastdims
end

reshape_cell_output(h, x) = reshape(h, :, size(x)[2:end]...)

# Stateful recurrence

"""
    Recur(cell)

`Recur` takes a recurrent cell and makes it stateful, managing the hidden state
in the background. `cell` should be a model of the form:

    h, y = cell(h, x...)

For example, here's a recurrent network that keeps a running total of its inputs:

# Examples
```jldoctest
julia> accum(h, x) = (h + x, x)
accum (generic function with 1 method)

julia> rnn = Flux.Recur(accum, 0)
Recur(accum)

julia> rnn(2) 
2

julia> rnn(3)
3

julia> rnn.state
5
```

Folding over a 3d Array of dimensions `(features, batch, time)` is also supported:

```jldoctest
julia> accum(h, x) = (h .+ x, x)
accum (generic function with 1 method)

julia> rnn = Flux.Recur(accum, zeros(Int, 1, 1))
Recur(accum)

julia> rnn([2])
1-element Vector{Int64}:
 2

julia> rnn([3])
1-element Vector{Int64}:
 3

julia> rnn.state
1×1 Matrix{Int64}:
 5

julia> out = rnn(reshape(1:10, 1, 1, :));  # apply to a sequence of (features, batch, time)

julia> out |> size
(1, 1, 10)

julia> vec(out)
10-element Vector{Int64}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10

julia> rnn.state
1×1 Matrix{Int64}:
 60
```
"""
mutable struct Recur{T,S} <: ContainerLayer
  cell::T
  state::S
end

function (m::Recur)(x)
  m.state, y = m.cell(m.state, x)
  return y
end

@functor Recur
trainable(a::Recur) = (; cell = a.cell)  # can't use <: PartialTrainLayer

Base.show(io::IO, m::Recur) = print(io, "Recur(", m.cell, ")")

"""
    reset!(rnn)

Reset the hidden state of a recurrent layer back to its original value.

Assuming you have a `Recur` layer `rnn`, this is roughly equivalent to:

    rnn.state = hidden(rnn.cell)

# Examples
```jldoctest
julia> r = Flux.RNNCell(relu, ones(1,1), zeros(1,1), ones(1,1), zeros(1,1));  # users should use the RNN wrapper struct instead

julia> y = Flux.Recur(r, ones(1,1));

julia> y.state
1×1 Matrix{Float64}:
 1.0

julia> y(ones(1,1))  # relu(1*1 + 1)
1×1 Matrix{Float64}:
 2.0

julia> y.state
1×1 Matrix{Float64}:
 2.0

julia> Flux.reset!(y)
1×1 Matrix{Float64}:
 0.0

julia> y.state
1×1 Matrix{Float64}:
 0.0
```
"""
reset!(m::Recur) = (m.state = m.cell.state0)
reset!(m) = foreach(reset!, functor(m)[1])

flip(f, xs) = reverse([f(x) for x in reverse(xs)])

function (m::Recur)(x::AbstractArray{T, 3}) where T
  h = [m(x_t) for x_t in eachlastdim(x)]
  sze = size(h[1])
  reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

# Vanilla RNN

struct RNNCell{F,A,V,S} <: SimpleLayer  # or should it be PartialTrainLayer{(:Wi, :Wh, :b)}?
  σ::F
  Wi::A
  Wh::A
  b::V
  state0::S
end

RNNCell((in, out)::Pair, σ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32) =
  RNNCell(σ, init(out, in), init(out, out), initb(out), init_state(out,1))

function (m::RNNCell{F,A,V,<:AbstractMatrix{T}})(h, x::Union{AbstractVecOrMat{T},OneHotArray}) where {F,A,V,T}
  Wi, Wh, b = m.Wi, m.Wh, m.b
  σ = NNlib.fast_act(m.σ, x)
  h = σ.(Wi*x .+ Wh*h .+ b)
  return h, reshape_cell_output(h, x)
end

@functor RNNCell

function Base.show(io::IO, l::RNNCell)
  print(io, "RNNCell(", size(l.Wi, 2), " => ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    RNN(in => out, σ = tanh)

The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.

The arguments `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(RNNCell(a...))`, and so RNNs are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

# Examples
```jldoctest
julia> r = RNN(3 => 5)
Recur(
  RNNCell(3 => 5, tanh),                # 50 parameters
)         # Total: 4 trainable arrays, 50 parameters,
          # plus 1 non-trainable, 5 parameters, summarysize 432 bytes.

julia> r(rand(Float32, 3)) |> size
(5,)

julia> Flux.reset!(r);

julia> r(rand(Float32, 3, 10)) |> size # batch size of 10
(5, 10)
```

!!! warning "Batch size changes"
  
    Failing to call `reset!` when the input batch size changes can lead to unexpected behavior. See the following example:

    ```julia
    julia> r = RNN(3 => 5)
    Recur(
      RNNCell(3 => 5, tanh),                # 50 parameters
    )         # Total: 4 trainable arrays, 50 parameters,
              # plus 1 non-trainable, 5 parameters, summarysize 432 bytes.

    julia> r.state |> size
    (5, 1)

    julia> r(rand(Float32, 3)) |> size
    (5,)

    julia> r.state |> size
    (5, 1)

    julia> r(rand(Float32, 3, 10)) |> size # batch size of 10
    (5, 10)

    julia> r.state |> size # state shape has changed
    (5, 10)

    julia> r(rand(Float32, 3)) |> size # erroneously outputs a length 5*10 = 50 vector.
    (50,)
    ```
"""
RNN(a...; ka...) = Recur(RNNCell(a...; ka...))
Recur(m::RNNCell) = Recur(m, m.state0)

# LSTM

struct LSTMCell{A,V,S} <: SimpleLayer  # or should it be PartialTrainLayer{(:Wi, :Wh, :b)}?
  Wi::A
  Wh::A
  b::V
  state0::S
end

function LSTMCell((in, out)::Pair;
                  init = glorot_uniform,
                  initb = zeros32,
                  init_state = zeros32)
  cell = LSTMCell(init(out * 4, in), init(out * 4, out), initb(out * 4), (init_state(out,1), init_state(out,1)))
  cell.b[gate(out, 2)] .= 1
  return cell
end

function (m::LSTMCell{A,V,<:NTuple{2,AbstractMatrix{T}}})((h, c), x::Union{AbstractVecOrMat{T},OneHotArray}) where {A,V,T}
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input, forget, cell, output = multigate(g, o, Val(4))
  c′ = @. sigmoid_fast(forget) * c + sigmoid_fast(input) * tanh_fast(cell)
  h′ = @. sigmoid_fast(output) * tanh_fast(c′)
  return (h′, c′), reshape_cell_output(h′, x)
end

@functor LSTMCell

Base.show(io::IO, l::LSTMCell) =
  print(io, "LSTMCell(", size(l.Wi, 2), " => ", size(l.Wi, 1)÷4, ")")

"""
    LSTM(in => out)

[Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
recurrent layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.

The arguments `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(LSTMCell(a...))`, and so LSTMs are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.

# Examples
```jldoctest
julia> l = LSTM(3 => 5)
Recur(
  LSTMCell(3 => 5),                     # 190 parameters
)         # Total: 5 trainable arrays, 190 parameters,
          # plus 2 non-trainable, 10 parameters, summarysize 1.062 KiB.

julia> l(rand(Float32, 3)) |> size
(5,)

julia> Flux.reset!(l);

julia> l(rand(Float32, 3, 10)) |> size # batch size of 10
(5, 10)
```

!!! warning "Batch size changes"
    Failing to call `reset!` when the input batch size changes can lead to unexpected behavior. See the example in [`RNN`](@ref).
"""
LSTM(a...; ka...) = Recur(LSTMCell(a...; ka...))
Recur(m::LSTMCell) = Recur(m, m.state0)

# GRU

function _gru_output(gxs, ghs, bs)
  r = @. sigmoid_fast(gxs[1] + ghs[1] + bs[1])
  z = @. sigmoid_fast(gxs[2] + ghs[2] + bs[2])
  return r, z
end

struct GRUCell{A,V,S} <: SimpleLayer  # or should it be PartialTrainLayer{(:Wi, :Wh, :b)}?
  Wi::A
  Wh::A
  b::V
  state0::S
end

GRUCell((in, out)::Pair; init = glorot_uniform, initb = zeros32, init_state = zeros32) =
  GRUCell(init(out * 3, in), init(out * 3, out), initb(out * 3), init_state(out,1))

function (m::GRUCell{A,V,<:AbstractMatrix{T}})(h, x::Union{AbstractVecOrMat{T},OneHotArray}) where {A,V,T}
  Wi, Wh, b, o = m.Wi, m.Wh, m.b, size(h, 1)
  gxs, ghs, bs = multigate(Wi*x, o, Val(3)), multigate(Wh*h, o, Val(3)), multigate(b, o, Val(3))
  r, z = _gru_output(gxs, ghs, bs)
  h̃ = @. tanh_fast(gxs[3] + r * ghs[3] + bs[3])
  h′ = @. (1 - z) * h̃ + z * h
  return h′, reshape_cell_output(h′, x)
end

@functor GRUCell

Base.show(io::IO, l::GRUCell) =
  print(io, "GRUCell(", size(l.Wi, 2), " => ", size(l.Wi, 1)÷3, ")")

"""
    GRU(in => out)

[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078v1) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences. This implements
the variant proposed in v1 of the referenced paper.

The integer arguments `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(GRUCell(a...))`, and so GRUs are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.

# Examples
```jldoctest
julia> g = GRU(3 => 5)
Recur(
  GRUCell(3 => 5),                      # 140 parameters
)         # Total: 4 trainable arrays, 140 parameters,
          # plus 1 non-trainable, 5 parameters, summarysize 792 bytes.

julia> g(rand(Float32, 3)) |> size
(5,)

julia> Flux.reset!(g);

julia> g(rand(Float32, 3, 10)) |> size # batch size of 10
(5, 10)
```

!!! warning "Batch size changes"
    Failing to call `reset!` when the input batch size changes can lead to unexpected behavior. See the example in [`RNN`](@ref).
"""
GRU(a...; ka...) = Recur(GRUCell(a...; ka...))
Recur(m::GRUCell) = Recur(m, m.state0)

# GRU v3

struct GRUv3Cell{A,V,S}
  Wi::A
  Wh::A
  b::V
  Wh_h̃::A
  state0::S
end

GRUv3Cell((in, out)::Pair; init = glorot_uniform, initb = zeros32, init_state = zeros32) =
  GRUv3Cell(init(out * 3, in), init(out * 2, out), initb(out * 3),
            init(out, out), init_state(out,1))

function (m::GRUv3Cell{A,V,<:AbstractMatrix{T}})(h, x::Union{AbstractVecOrMat{T},OneHotArray}) where {A,V,T}
  Wi, Wh, b, Wh_h̃, o = m.Wi, m.Wh, m.b, m.Wh_h̃, size(h, 1)
  gxs, ghs, bs = multigate(Wi*x, o, Val(3)), multigate(Wh*h, o, Val(2)), multigate(b, o, Val(3))
  r, z = _gru_output(gxs, ghs, bs)
  h̃ = tanh_fast.(gxs[3] .+ (Wh_h̃ * (r .* h)) .+ bs[3])
  h′ = @. (1 - z) * h̃ + z * h
  return h′, reshape_cell_output(h′, x)
end

@functor GRUv3Cell

Base.show(io::IO, l::GRUv3Cell) =
  print(io, "GRUv3Cell(", size(l.Wi, 2), " => ", size(l.Wi, 1)÷3, ")")

"""
    GRUv3(in => out)

[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078v3) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences. This implements
the variant proposed in v3 of the referenced paper.

The arguments `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(GRUv3Cell(a...))`, and so GRUv3s are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.

# Examples
```jldoctest
julia> g = GRUv3(3 => 5)
Recur(
  GRUv3Cell(3 => 5),                    # 140 parameters
)         # Total: 5 trainable arrays, 140 parameters,
          # plus 1 non-trainable, 5 parameters, summarysize 848 bytes.

julia> g(rand(Float32, 3)) |> size
(5,)

julia> Flux.reset!(g);

julia> g(rand(Float32, 3, 10)) |> size # batch size of 10
(5, 10)
```

!!! warning "Batch size changes"
    Failing to call `reset!` when the input batch size changes can lead to unexpected behavior. See the example in [`RNN`](@ref).
"""
GRUv3(a...; ka...) = Recur(GRUv3Cell(a...; ka...))
Recur(m::GRUv3Cell) = Recur(m, m.state0)
