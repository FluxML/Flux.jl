
gate(h, n) = (1:h) .+ h*(n-1)
gate(x::AbstractVector, h, n) = @view x[gate(h,n)]
gate(x::AbstractMatrix, h, n) = view(x, gate(h,n), :)

# AD-friendly helper for dividing monolithic RNN params into equally sized gates
multigate(x::AbstractArray, h, ::Val{N}) where N = ntuple(n -> gate(x,h,n), N)

function ChainRulesCore.rrule(::typeof(multigate), x::AbstractArray, h, c)
  function multigate_pullback(dy)
    dx = map!(zero, similar(x, float(eltype(x)), axes(x)), x)
    foreach(multigate(dx, h, c), unthunk(dy)) do dxᵢ, dyᵢ
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



# non-stateful recurrence

"""
  scan_full

Recreating jax.lax.scan functionality in julia. Takes a function, initial carry and a sequence, then returns the full output of the sequence and the final carry. See `scan_partial` to only return the final output of the sequence. 
"""
function scan_full(func, init_carry, xs::AbstractVector{<:AbstractArray})
  # Recurrence operation used in the fold. Takes the state of the
  # fold and the next input, returns the new state.
  function recurrence_op((carry, outputs), input)
    carry, out = func(carry, input)
    return carry, vcat(outputs, [out])
  end
  # Fold left to right.
  return Base.mapfoldl_impl(identity, recurrence_op, (init_carry, empty(xs)), xs)
end

function scan_full(func, init_carry, x_block)
  # x_block is an abstractarray and we want to scan over the last dimension.
  xs_ = Flux.eachlastdim(x_block)

  # this is needed due to a bug in eachlastdim which produces a vector in a
  # gradient context, but a generator otherwise.
  xs = if xs_ isa Base.Generator
    collect(xs_) # eachlastdim produces a generator in non-gradient environment
  else
    xs_
  end
  scan_full(func, init_carry, xs)
end

# Chain Rule for Base.mapfoldl_impl
function ChainRulesCore.rrule(
  config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
  ::typeof(Base.mapfoldl_impl),
  ::typeof(identity),
  op::G,
  init,
  x::Union{AbstractArray, Tuple};
) where {G}
  # Hobbits has two types afaict, first is for the first component, then the second component.
  # This has to do with the entrance I believe (i.e. we don't know what function enters, but we know what
  # function is called in subsequent things...
  # hobbits = Vector{Tuple}(undef, length(x))  # Unfornately Zygote needs this
  # accum_init = ChainRulesCore.rrule_via_ad(config, op, init[1], nothing)
  # @show typeof(accum_init)
  accum_init = ChainRulesCore.rrule_via_ad(config, op, init, x[1])
  hobbits = accumulate(x[begin+1:end]; init=accum_init) do (a, _), b
    @show a, b
    c, back = ChainRulesCore.rrule_via_ad(config, op, a, b)
  end
  # @show typeof(hobbits)

  y = first(last(hobbits))
  axe = axes(x)
  project = ChainRulesCore.ProjectTo(x)
  function unfoldl(dy)
    trio = accumulate(Iterators.reverse(hobbits); init=(0, dy, 0)) do (_, dc, _), (_, back)
      ds, da, db = back(dc)
    end
    # @show trio
    f_ds, f_da, f_db = accum_init[2](trio[end][2])
    dop = sum(first, trio) + f_ds
    dx = [[f_db]; map(last, Iterators.reverse(trio))]
    d_init = f_da
    return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), dop, d_init, project(reshape(dx, axe)))
  end
  return y, unfoldl
end

# function ChainRulesCore.rrule(
#   config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
#   ::typeof(Base.mapfoldl_impl),
#   ::typeof(identity),
#   op::G,
#   init,
#   x::Union{AbstractArray, Tuple};
# ) where {G}
#   hobbits = Vector{Any}(undef, length(x))  # Unfornately Zygote needs this
#   accumulate!(hobbits, x; init=(init, nothing)) do (a, _), b
#     c, back = ChainRulesCore.rrule_via_ad(config, op, a, b)
#   end
#   y = first(last(hobbits))
#   axe = axes(x)
#   project = ChainRulesCore.ProjectTo(x)
#   function unfoldl(dy)
#     trio = accumulate(Iterators.reverse(hobbits); init=(0, dy, 0)) do (_, dc, _), (_, back)
#       ds, da, db = back(dc)
#     end
#     dop = sum(first, trio)
#     dx = map(last, Iterators.reverse(trio))
#     @show dx
#     d_init = trio[end][2]
#     return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), dop, d_init, project(reshape(dx, axe)))
#   end
#   return y, unfoldl
# end


"""
  scan_partial

Recreating jax.lax.scan functionality in julia. Takes a function, initial carry and a sequence, then returns the final output of the sequence and the final carry. See `scan_full` to return the entire output sequence.
"""
function scan_partial(func, init_carry, xs::AbstractVector{<:AbstractArray})
  x_init, x_rest = Iterators.peel(xs)
  (carry, y) = func(init_carry, x_init)
  for x in x_rest
    (carry, y) = func(carry, x)
  end
  carry, y
end

function scan_partial(func, init_carry, x_block)
  # x_block is an abstractarray and we want to scan over the last dimension.
  xs_ = Flux.eachlastdim(x_block)
  
  # this is needed due to a bug in eachlastdim which produces a vector in a
  # gradient context, but a generator otherwise.
  xs = if xs_ isa Base.Generator
    collect(xs_) # eachlastdim produces a generator in non-gradient environment
  else
    xs_
  end
  scan_partial(func, init_carry, xs)
end


"""
  NewRecur
New Recur. An experimental recur interface for removing statefullness in recurrent architectures for flux. This struct has two type parameters. The first `RET_SEQUENCE` is a boolean which determines whether `scan_full` (`RET_SEQUENCE=true`) or `scan_partial` (`RET_SEQUENCE=false`) is used to scan through the sequence. This structure has no internal state, and instead returns:

```julia
l = NewRNN(1,2)
xs # Some input array Input x BatchSize x Time
init_carry # the initial carry of the cell.
l(xs) # -> returns the output of the RNN, uses cell.state0 as init_carry.
l(init_carry, xs) # -> returns (final_carry, output), where the size ofoutput is determined by RET_SEQUENCE.
```
"""
struct NewRecur{RET_SEQUENCE, T}
  cell::T
  # state::S
  function NewRecur(cell; return_sequence::Bool=false)
    new{return_sequence, typeof(cell)}(cell)
  end
  function NewRecur{true}(cell)
    new{true, typeof(cell)}(cell)
  end
  function NewRecur{false}(cell)
    new{false, typeof(cell)}(cell)
  end
end

Flux.@functor NewRecur
Flux.trainable(a::NewRecur) = (; cell = a.cell)
Base.show(io::IO, m::NewRecur) = print(io, "NewRecur(", m.cell, ")")

(l::NewRecur)(init_carry, x_mat::AbstractMatrix) = MethodError("Matrix is ambiguous with NewRecur")
(l::NewRecur)(init_carry, x_mat::AbstractVector{T}) where {T<:Number} = MethodError("Vector is ambiguous with NewRecur")

function (l::NewRecur)(xs::AbstractArray)
  results = l(l.cell.state0, xs)
  results[2] # Only return the output here.
end

function (l::NewRecur{false})(init_carry, xs)
  results = scan_partial(l.cell, init_carry, xs)
  results[1], results[2]
end

function (l::NewRecur{true})(init_carry, xs)
  results = scan_full(l.cell, init_carry, xs)
  results[1], stack(results[2], dims=3)
end



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
mutable struct Recur{T,S}
  cell::T
  state::S
end

function (m::Recur)(x)
  m.state, y = m.cell(m.state, x)
  return y
end

@functor Recur
trainable(a::Recur) = (; cell = a.cell)

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


########
#
# Recurrent Cells
#
########

# Vanilla RNN
struct RNNCell{F,I,H,V,S}
  σ::F
  Wi::I
  Wh::H
  b::V
  state0::S
end

RNNCell((in, out)::Pair, σ=tanh; init=Flux.glorot_uniform, initb=zeros32, init_state=zeros32) =
  RNNCell(σ, init(out, in), init(out, out), initb(out), init_state(out,1))

function (m::RNNCell{F,I,H,V,<:AbstractMatrix{T}})(h, x::AbstractVecOrMat) where {F,I,H,V,T}
  Wi, Wh, b = m.Wi, m.Wh, m.b
  _size_check(m, x, 1 => size(Wi,2))
  σ = NNlib.fast_act(m.σ, x)
  xT = _match_eltype(m, T, x)
  h = σ.(Wi*xT .+ Wh*h .+ b)
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

# Note:
`RNNCell`s can be constructed directly by specifying the non-linear function, the `Wi` and `Wh` internal matrices, a bias vector `b`, and a learnable initial state `state0`. The  `Wi` and `Wh` matrices do not need to be the same type, but if `Wh` is `dxd`, then `Wi` should be of shape `dxN`.

```julia
julia> using LinearAlgebra

julia> r = Flux.Recur(Flux.RNNCell(tanh, rand(5, 4), Tridiagonal(rand(5, 5)), rand(5), rand(5, 1)))

julia> r(rand(4, 10)) |> size # batch size of 10
(5, 10)
```
"""
RNN(a...; ka...) = Recur(RNNCell(a...; ka...))
Recur(m::RNNCell) = Recur(m, m.state0)

NewRNN(a...; return_sequence::Bool=false, ka...) = NewRecur(Flux.RNNCell(a...; ka...); return_sequence=return_sequence)

# LSTM

struct LSTMCell{I,H,V,S}
  Wi::I
  Wh::H
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

function (m::LSTMCell{I,H,V,<:NTuple{2,AbstractMatrix{T}}})((h, c), x::AbstractVecOrMat) where {I,H,V,T}
  _size_check(m, x, 1 => size(m.Wi,2))
  b, o = m.b, size(h, 1)
  xT = _match_eltype(m, T, x)
  g = muladd(m.Wi, xT, muladd(m.Wh, h, b))
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

# Note:
  `LSTMCell`s can be constructed directly by specifying the non-linear function, the `Wi` and `Wh` internal matrices, a bias vector `b`, and a learnable initial state `state0`. The  `Wi` and `Wh` matrices do not need to be the same type. See the example in [`RNN`](@ref).
"""
LSTM(a...; ka...) = Recur(LSTMCell(a...; ka...))
Recur(m::LSTMCell) = Recur(m, m.state0)

NewLSTM(a...; return_sequence::Bool=false, ka...) = NewRecur(Flux.LSTMCell(a...; ka...); return_sequence=return_sequence)

# GRU

function _gru_output(gxs, ghs, bs)
  r = @. sigmoid_fast(gxs[1] + ghs[1] + bs[1])
  z = @. sigmoid_fast(gxs[2] + ghs[2] + bs[2])
  return r, z
end

struct GRUCell{I,H,V,S}
  Wi::I
  Wh::H
  b::V
  state0::S
end

GRUCell((in, out)::Pair; init = glorot_uniform, initb = zeros32, init_state = zeros32) =
  GRUCell(init(out * 3, in), init(out * 3, out), initb(out * 3), init_state(out,1))

function (m::GRUCell{I,H,V,<:AbstractMatrix{T}})(h, x::AbstractVecOrMat) where {I,H,V,T}
  _size_check(m, x, 1 => size(m.Wi,2))
  Wi, Wh, b, o = m.Wi, m.Wh, m.b, size(h, 1)
  xT = _match_eltype(m, T, x)
  gxs, ghs, bs = multigate(Wi*xT, o, Val(3)), multigate(Wh*h, o, Val(3)), multigate(b, o, Val(3))
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

# Note:
  `GRUCell`s can be constructed directly by specifying the non-linear function, the `Wi` and `Wh` internal matrices, a bias vector `b`, and a learnable initial state `state0`. The  `Wi` and `Wh` matrices do not need to be the same type. See the example in [`RNN`](@ref).
"""
GRU(a...; ka...) = Recur(GRUCell(a...; ka...))
Recur(m::GRUCell) = Recur(m, m.state0)

NewGRU(a...; return_sequence::Bool=false, ka...) = NewRecur(Flux.GRUCell(a...; ka...); return_sequence=return_sequence)

# GRU v3

struct GRUv3Cell{I,H,V,HH,S}
  Wi::I
  Wh::H
  b::V
  Wh_h̃::HH
  state0::S
end

GRUv3Cell((in, out)::Pair; init = glorot_uniform, initb = zeros32, init_state = zeros32) =
  GRUv3Cell(init(out * 3, in), init(out * 2, out), initb(out * 3),
            init(out, out), init_state(out,1))

function (m::GRUv3Cell{I,H,V,HH,<:AbstractMatrix{T}})(h, x::AbstractVecOrMat) where {I,H,V,HH,T}
  _size_check(m, x, 1 => size(m.Wi,2))
  Wi, Wh, b, Wh_h̃, o = m.Wi, m.Wh, m.b, m.Wh_h̃, size(h, 1)
  xT = _match_eltype(m, T, x)
  gxs, ghs, bs = multigate(Wi*xT, o, Val(3)), multigate(Wh*h, o, Val(2)), multigate(b, o, Val(3))
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

# Note:
  `GRUv3Cell`s can be constructed directly by specifying the non-linear function, the `Wi`, `Wh`, and `Wh_h` internal matrices, a bias vector `b`, and a learnable initial state `state0`. The  `Wi`, `Wh`, and `Wh_h` matrices do not need to be the same type. See the example in [`RNN`](@ref).
"""
GRUv3(a...; ka...) = Recur(GRUv3Cell(a...; ka...))
Recur(m::GRUv3Cell) = Recur(m, m.state0)

NewGRUv3(a...; return_sequence::Bool=false, ka...) = NewRecur(Flux.GRUv3Cell(a...; ka...); return_sequence=return_sequence)
