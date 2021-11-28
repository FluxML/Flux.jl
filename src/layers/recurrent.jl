
gate(h, n) = (1:h) .+ h*(n-1)
gate(x::AbstractVector, h, n) = @view x[gate(h,n)]
gate(x::AbstractMatrix, h, n) = x[gate(h,n),:]

# Stateful recurrence

"""
    Recur(cell)

`Recur` takes a recurrent cell and makes it stateful, managing the hidden state
in the background. `cell` should be a model of the form:

    h, y = cell(h, x...)

For example, here's a recurrent network that keeps a running total of its inputs:

```julia
accum(h, x) = (h + x, x)
rnn = Flux.Recur(accum, 0)
rnn(2)      # 2
rnn(3)      # 3
rnn.state   # 5
rnn.(1:10)  # apply to a sequence
rnn.state   # 60
```

Folding over a 3d Array of dimensions `(features, batch, time)` is also supported:

```julia
accum(h, x) = (h .+ x, x)
rnn = Flux.Recur(accum, zeros(Int, 1, 1))
rnn([2])                    # 2
rnn([3])                    # 3
rnn.state                   # 5
rnn(reshape(1:10, 1, 1, :)) # apply to a sequence of (features, batch, time)
rnn.state                   # 60
```

"""
mutable struct Recur{T,S}
  cell::T
  state::S
end

function (m::Recur)(x)
  m.state, y = m.cell(m.state, x)
  return y
end

@functor Recur
trainable(a::Recur) = (a.cell,)

Base.show(io::IO, m::Recur) = print(io, "Recur(", m.cell, ")")

"""
    reset!(rnn)

Reset the hidden state of a recurrent layer back to its original value.

Assuming you have a `Recur` layer `rnn`, this is roughly equivalent to:
```julia
rnn.state = hidden(rnn.cell)
```
"""
reset!(m::Recur) = (m.state = m.cell.state0)
reset!(m) = foreach(reset!, functor(m)[1])


# TODO remove in v0.13
function Base.getproperty(m::Recur, sym::Symbol)
  if sym === :init
    Zygote.ignore() do
      @warn "Recur field :init has been deprecated. To access initial state weights, use m::Recur.cell.state0 instead."
    end
    return getfield(m.cell, :state0)
  else
    return getfield(m, sym)
  end
end

flip(f, xs) = reverse(f.(reverse(xs)))

function (m::Recur)(x::AbstractArray{T, 3}) where T
  h = [m(view(x, :, :, i)) for i in 1:size(x, 3)]
  sze = size(h[1])
  reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

# Vanilla RNN

struct RNNCell{F,A,V,S}
  σ::F
  Wi::A
  Wh::A
  b::V
  state0::S
end

RNNCell(in::Integer, out::Integer, σ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32) = 
  RNNCell(σ, init(out, in), init(out, out), initb(out), init_state(out,1))

function (m::RNNCell{F,A,V,<:AbstractMatrix{T}})(h, x::Union{AbstractVecOrMat{T},OneHotArray}) where {F,A,V,T}
  σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
  h = σ.(Wi*x .+ Wh*h .+ b)
  sz = size(x)
  return h, reshape(h, :, sz[2:end]...)
end

@functor RNNCell

function Base.show(io::IO, l::RNNCell)
  print(io, "RNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    RNN(in::Integer, out::Integer, σ = tanh)

The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.

The parameters `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(RNNCell(a...))`, and so RNNs are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

# Examples
```jldoctest
julia> r = RNN(3, 5)
Recur(
  RNNCell(3, 5, tanh),                  # 50 parameters
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
    julia> r = RNN(3, 5)
    Recur(
      RNNCell(3, 5, tanh),                  # 50 parameters
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

# TODO remove in v0.13
function Base.getproperty(m::RNNCell, sym::Symbol)
  if sym === :h
    Zygote.ignore() do
      @warn "RNNCell field :h has been deprecated. Use m::RNNCell.state0 instead."
    end
    return getfield(m, :state0)
  else
    return getfield(m, sym)
  end
end

# LSTM

struct LSTMCell{A,V,S}
  Wi::A
  Wh::A
  b::V
  state0::S
end

function LSTMCell(in::Integer, out::Integer;
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
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  sz = size(x)
  return (h′, c), reshape(h′, :, sz[2:end]...)
end

@functor LSTMCell

Base.show(io::IO, l::LSTMCell) =
  print(io, "LSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")

"""
    LSTM(in::Integer, out::Integer)

[Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
recurrent layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.

The parameters `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(LSTMCell(a...))`, and so LSTMs are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.

# Examples
```jldoctest
julia> l = LSTM(3, 5)
Recur(
  LSTMCell(3, 5),                       # 190 parameters
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

# TODO remove in v0.13
function Base.getproperty(m::LSTMCell, sym::Symbol)
  if sym === :h
    Zygote.ignore() do
      @warn "LSTMCell field :h has been deprecated. Use m::LSTMCell.state0[1] instead."
    end
    return getfield(m, :state0)[1]
  elseif sym === :c
    Zygote.ignore() do
      @warn "LSTMCell field :c has been deprecated. Use m::LSTMCell.state0[2] instead."
    end  
    return getfield(m, :state0)[2]
  else
    return getfield(m, sym)
  end
end

# GRU

function _gru_output(Wi, Wh, b, x, h)
  o = size(h, 1)
  gx, gh = Wi*x, Wh*h
  r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
  z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))

  return gx, gh, r, z
end

struct GRUCell{A,V,S}
  Wi::A
  Wh::A
  b::V
  state0::S
end

GRUCell(in, out; init = glorot_uniform, initb = zeros32, init_state = zeros32) =
  GRUCell(init(out * 3, in), init(out * 3, out), initb(out * 3), init_state(out,1))

function (m::GRUCell{A,V,<:AbstractMatrix{T}})(h, x::Union{AbstractVecOrMat{T},OneHotArray}) where {A,V,T}
  b, o = m.b, size(h, 1)
  gx, gh, r, z = _gru_output(m.Wi, m.Wh, b, x, h)
  h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
  h′ = (1 .- z) .* h̃ .+ z .* h
  sz = size(x)
  return h′, reshape(h′, :, sz[2:end]...)
end

@functor GRUCell

Base.show(io::IO, l::GRUCell) =
  print(io, "GRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    GRU(in::Integer, out::Integer)

[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078v1) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences. This implements
the variant proposed in v1 of the referenced paper.

The parameters `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(GRUCell(a...))`, and so GRUs are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.

# Examples
```jldoctest
julia> g = GRU(3, 5)
Recur(
  GRUCell(3, 5),                        # 140 parameters
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

# TODO remove in v0.13
function Base.getproperty(m::GRUCell, sym::Symbol)
  if sym === :h
    Zygote.ignore() do
      @warn "GRUCell field :h has been deprecated. Use m::GRUCell.state0 instead."
    end
    return getfield(m, :state0)
  else
    return getfield(m, sym)
  end
end


# GRU v3

struct GRUv3Cell{A,V,S}
  Wi::A
  Wh::A
  b::V
  Wh_h̃::A
  state0::S
end

GRUv3Cell(in, out; init = glorot_uniform, initb = zeros32, init_state = zeros32) =
  GRUv3Cell(init(out * 3, in), init(out * 2, out), initb(out * 3), 
            init(out, out), init_state(out,1))

function (m::GRUv3Cell{A,V,<:AbstractMatrix{T}})(h, x::Union{AbstractVecOrMat{T},OneHotArray}) where {A,V,T}
  b, o = m.b, size(h, 1)
  gx, gh, r, z = _gru_output(m.Wi, m.Wh, b, x, h)
  h̃ = tanh.(gate(gx, o, 3) .+ (m.Wh_h̃ * (r .* h)) .+ gate(b, o, 3))
  h′ = (1 .- z) .* h̃ .+ z .* h
  sz = size(x)
  return h′, reshape(h′, :, sz[2:end]...)
end

@functor GRUv3Cell

Base.show(io::IO, l::GRUv3Cell) =
  print(io, "GRUv3Cell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    GRUv3(in::Integer, out::Integer)

[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078v3) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences. This implements
the variant proposed in v3 of the referenced paper.

The parameters `in` and `out` describe the size of the feature vectors passed as input and as output. That is, it accepts a vector of length `in` or a batch of vectors represented as a `in x B` matrix and outputs a vector of length `out` or a batch of vectors of size `out x B`.

This constructor is syntactic sugar for `Recur(GRUv3Cell(a...))`, and so GRUv3s are stateful. Note that the state shape can change depending on the inputs, and so it is good to `reset!` the model between inference calls if the batch size changes. See the examples below.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.

# Examples
```jldoctest
julia> g = GRUv3(3, 5)
Recur(
  GRUv3Cell(3, 5),                      # 140 parameters
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


@adjoint function Broadcast.broadcasted(f::Recur, args...)
  Zygote.∇map(__context__, f, args...)
end


"""
    Bidirectional{A,B} 

A wrapper layer that allows the use of [bidirectional](https://ieeexplore.ieee.org/document/650093) layers. It contains two parts that are Flux layers: `forward` and `backward` where 
the forward layer weights are concatenated with the reversed order of the backward layer weights.

It is intended to be used with recurrent layers such as `LSTM`, `GRU` or `RNN` to benefit from the sequential information that recurrent 
layers have, but it will not raise an error if used with a different layer such as `Dense`, as long as the layer is compatible with the concatenation function `vcat`.

# Examples
```jldoctest
julia> BLSTM = Bidirectional(LSTM, 3, 5)
Bidirectional(
  Recur(
    LSTMCell(3, 5),                     # 190 parameters
  ),
  Recur(
    LSTMCell(3, 5),                     # 190 parameters
  ),
)         # Total: 10 trainable arrays, 380 parameters,
          # plus 4 non-trainable, 20 parameters, summarysize 2.141 KiB.
julia> Bidirectional(LSTM, 3, 5)(rand(Float32, 3)) |> size
(10,)

julia> Bidirectional(LSTM, 3, 5)(rand(Float32, 3))
10-element Vector{Float32}:
  0.009660141
 -0.011628074
 -0.017348368
 -0.002131971
  0.0152327195
 -0.024785668
  0.021315152
 -0.015476399
  0.01589005
 -0.002576883

julia> model = Chain(Embedding(10000, 200), Bidirectional(LSTM, 200, 128), Dense(256, 5), softmax)
Chain(
  Embedding(10000, 200),                # 2_000_000 parameters
  Bidirectional(
    Recur(
      LSTMCell(200, 128),               # 168_704 parameters
    ),
    Recur(
      LSTMCell(200, 128),               # 168_704 parameters
    ),
  ),
  Dense(256, 5),                        # 1_285 parameters
  NNlib.softmax,
)         # Total: 13 trainable arrays, 2_338_693 parameters,
          # plus 4 non-trainable, 512 parameters, summarysize 8.922 MiB.
```
"""
struct Bidirectional{A,B} 
  forward::A
  backward::B
end

# Constructor that creates a bidirectional with the same layer for forward and backward
Bidirectional(rnn, a...; ka...) = Bidirectional(rnn(a...; ka...), rnn(a...; ka...))


# Concatenate the forward and reversed backward weights
function (m::Bidirectional)(x::Union{AbstractVecOrMat{T},OneHotArray}) where {T}
  return vcat(m.forward(x), reverse(m.backward(reverse(x; dims=1)); dims=1))
end

@functor Bidirectional
