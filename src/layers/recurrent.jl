
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



# Vanilla RNN

@doc raw"""
    RNNCell(in => out, σ = tanh; init = glorot_uniform, bias = true)

The most basic recurrent layer. Essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.

In the forward pass, implements the function
```math
h^\prime = \sigma(W_i x + W_h h + b)
```
and returns `h'`.

See also [`RNN`](@ref).

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `σ`: The non-linearity to apply to the output. Default is `tanh`.
- `init`: The initialization function to use for the weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    rnncell(x, h)

The arguments of the forward pass are:

- `x`: The input to the RNN. It should be a vector of size `in` or a matrix of size `in x batch_size`.
- `h`: The hidden state of the RNN. It should be a vector of size `out` or a matrix of size `out x batch_size`.

# Examples

```jldoctest
r = RNNCell(3 => 5)

# A sequence of length 10 and batch size 4
x = [rand(Float32, 3, 4) for _ in 1:10]

# Initialize the hidden state
h = zeros(Float32, 5)

# We collect the hidden states in an array `history`
# in case the loss depends on the entire sequence.
ŷ = []

for x_t in x
  h = r(x_t, h)
  ŷ = [ŷ..., h] # Cannot use `push!(ŷ, h)` here since mutation 
                # is not automatic differentiation friendly yet.
                # Can use `y = vcat(y, [h])` as an alternative.
end

h   # The final hidden state
ŷ   # The hidden states at each time step
```
"""
struct RNNCell{F,I,H,V}
  σ::F
  Wi::I
  Wh::H
  bias::V
end

@layer RNNCell 

function RNNCell((in, out)::Pair, σ=tanh; init = glorot_uniform, bias = true)
  Wi = init(out, in)
  Wh = init(out, out)
  b = create_bias(Wi, bias, size(Wi, 1))
  return RNNCell(σ, Wi, Wh, b)
end

function (m::RNNCell)(x::AbstractVecOrMat, h::AbstractVecOrMat)
  _size_check(m, x, 1 => size(m.Wi,2))
  σ = NNlib.fast_act(m.σ, x)
  h = σ.(m.Wi*x .+ m.Wh*h .+ m.bias)
  return h
end

function Base.show(io::IO, l::RNNCell)
  print(io, "RNNCell(", size(l.Wi, 2), " => ", size(l.Wi, 1))
  print(io, ", ", l.σ)
  print(io, ")")
end

@doc raw"""
    RNN(in => out, σ = tanh; bias = true, init = glorot_uniform)

The most basic recurrent layer. Essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.  

In the forward pass computes

```math
h_t = \sigma(W_i x_t + W_h h_{t-1} + b)
```
for all `len` steps `t` in the in input sequence. 
Returns all hidden states `h_t` in a tensor of size `(out, len, batch_size)`.

See also [`RNNCell`](@ref).

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `σ`: The non-linearity to apply to the output. Default is `tanh`.
- `init`: The initialization function to use for the weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    rnn(x, h)

The arguments of the forward pass are:

- `x`: The input to the RNN. It should be a matrix size `in x len` or a tensor of size `in x len x batch_size`.
- `h`: The hidden state of the RNN. It should be a vector of size `out` or a matrix of size `out x batch_size`.

# Examples

```jldoctest
julia> d_in, d_out, len, batch_size = 4, 6, 3, 5;

julia> x = rand(Float32, (d_in, len, batch_size));

julia> h = zeros(Float32, (d_out, batch_size));

julia> rnn = RNN(d_in => d_out)
RNN(
  RNNCell(4 => 6, tanh),                # 66 parameters
)                   # Total: 3 arrays, 66 parameters, 424 bytes.

julia> y = rnn(x, h);   # [y] = [d_out, len, batch_size]
```

Sometimes, the initial hidden state is a learnable parameter. 
In this case, the `RNN` should be wrapped in a custom struct.

```jldoctest
struct Model
  rnn::RNN
  h0::AbstractVector
end

Flux.@layer :expand Model

(m::Model)(x) = m.rnn(x, m.h0)

model = Model(RNN(32 => 64), zeros(Float32, 64))
```
"""
struct RNN{M}
  cell::M
end

@layer :expand RNN

function RNN((in, out)::Pair, σ = tanh; bias = true, init = glorot_uniform)
  cell = RNNCell(in => out, σ; bias, init)
  return RNN(cell)
end

function (m::RNN)(x, h) 
  @assert ndims(x) == 2 || ndims(x) == 3
  # [x] = [in, L] or [in, L, B]
  # [h] = [out] or [out, B]
  y = []
  for x_t in eachslice(x, dims=2)
    h = m.cell(x_t, h)
    # y = [y..., h]
    y = vcat(y, [h])
  end
  return stack(y, dims=2)
end


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

@layer LSTMCell

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
          # plus 2 non-trainable, 10 parameters, summarysize 1.023 KiB.

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

@layer GRUCell

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
          # plus 1 non-trainable, 5 parameters, summarysize 784 bytes.

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

@layer GRUv3Cell

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
          # plus 1 non-trainable, 5 parameters, summarysize 840 bytes.

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
