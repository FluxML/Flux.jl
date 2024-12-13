out_from_state(state) = state
out_from_state(state::Tuple) = state[1]

function scan(cell, x, state)
  y = []
  for x_t in eachslice(x, dims = 2)
    state = cell(x_t, state)
    out = out_from_state(state)
    y = vcat(y, [out])
  end
  return stack(y, dims = 2)
end

"""
    Recurrence(cell)

Create a recurrent layer that processes entire sequences out
of a recurrent `cell`, such as an [`RNNCell`](@ref), [`LSTMCell`](@ref), or [`GRUCell`](@ref),
similarly to how [`RNN`](@ref), [`LSTM`](@ref), and [`GRU`](@ref) process sequences.

The `cell` should be a callable object that takes an input `x` and a hidden state `state` and returns
a new hidden state `state'`. The `cell` should also implement the `initialstates` method that returns
the initial hidden state. The output of the `cell` is considered to be:
1. The first element of the `state` tuple if `state` is a tuple (e.g. `(h, c)` for LSTM).
2. The `state` itself if `state` is not a tuple, e.g. an array `h` for RNN and GRU.

# Forward

    rnn(x, [state])

The input `x` should be an array of size `in x len` or `in x len x batch_size`, 
where `in` is the input dimension of the cell, `len` is the sequence length, and `batch_size` is the batch size.
The `state` should be a valid state for the recurrent cell. If not provided, it obtained by calling
`Flux.initialstates(cell)`.

The output is an array of size `out x len x batch_size`, where `out` is the output dimension of the cell.

The operation performed is semantically equivalent to the following code:
```julia
out_from_state(state) = state
out_from_state(state::Tuple) = state[1]

state = Flux.initialstates(cell)
out = []
for x_t in eachslice(x, dims = 2)
  state = cell(x_t, state)
  out = [out..., out_from_state(state)]
end
stack(out, dims = 2)
```

# Examples

```jldoctest
julia> rnn = Recurrence(RNNCell(2 => 3))

julia> x = rand(Float32, 2, 3, 4); # in x len x batch_size

julia> y = rnn(x); # out x len x batch_size
```
"""
struct Recurrence{M}
  cell::M
end

@layer Recurrence

initialstates(rnn::Recurrence) = initialstates(rnn.cell)

(rnn::Recurrence)(x::AbstractArray) = rnn(x, initialstates(rnn))
(rnn::Recurrence)(x::AbstractArray, state) = scan(rnn.cell, x, state)

# Vanilla RNN
@doc raw"""
    RNNCell(in => out, σ = tanh; init_kernel = glorot_uniform, 
      init_recurrent_kernel = glorot_uniform, bias = true)

The most basic recurrent layer. Essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.

In the forward pass, implements the function
```math
h^\prime = \sigma(W_i x + W_h h + b)
```
and returns `h'`.

See [`RNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `σ`: The non-linearity to apply to the output. Default is `tanh`.
- `init_kernel`: The initialization function to use for the input to hidden connection weights. Default is `glorot_uniform`.
- `init_recurrent_kernel`: The initialization function to use for the hidden to hidden connection weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    rnncell(x, [h])

The arguments of the forward pass are:

- `x`: The input to the RNN. It should be a vector of size `in` or a matrix of size `in x batch_size`.
- `h`: The hidden state of the RNN. It should be a vector of size `out` or a matrix of size `out x batch_size`.
       If not provided, it is assumed to be a vector of zeros, initialized by [`initialstates`](@ref).

# Examples

```julia
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
struct RNNCell{F, I, H, V}
  σ::F
  Wi::I
  Wh::H
  bias::V
end

@layer RNNCell

"""
    initialstates(rnn) -> AbstractVector

Return the initial hidden state for the given recurrent cell or recurrent layer.

# Example
```julia
using Flux

# Create an RNNCell from input dimension 10 to output dimension 20
rnn = RNNCell(10 => 20)

# Get the initial hidden state
h0 = initialstates(rnn)

# Get some input data
x = rand(Float32, 10)

# Run forward
res = rnn(x, h0)
"""
initialstates(rnn::RNNCell) = zeros_like(rnn.Wh, size(rnn.Wh, 2))

function RNNCell(
  (in, out)::Pair,
  σ = tanh;
  init_kernel = glorot_uniform,
  init_recurrent_kernel = glorot_uniform,
  bias = true,
)
  Wi = init_kernel(out, in)
  Wh = init_recurrent_kernel(out, out)
  b = create_bias(Wi, bias, size(Wi, 1))
  return RNNCell(σ, Wi, Wh, b)
end

function (rnn::RNNCell)(x::AbstractVecOrMat)
  state = initialstates(rnn)
  return rnn(x, state)
end

function (m::RNNCell)(x::AbstractVecOrMat, h::AbstractVecOrMat)
  _size_check(m, x, 1 => size(m.Wi, 2))
  σ = NNlib.fast_act(m.σ, x)
  h = σ.(m.Wi * x .+ m.Wh * h .+ m.bias)
  return h
end

function Base.show(io::IO, m::RNNCell)
  print(io, "RNNCell(", size(m.Wi, 2), " => ", size(m.Wi, 1))
  print(io, ", ", m.σ)
  print(io, ")")
end

@doc raw"""
    RNN(in => out, σ = tanh; init_kernel = glorot_uniform, 
      init_recurrent_kernel = glorot_uniform, bias = true)

The most basic recurrent layer. Essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.  

In the forward pass computes

```math
h_t = \sigma(W_i x_t + W_h h_{t-1} + b)
```
for all `len` steps `t` in the in input sequence. 

See [`RNNCell`](@ref) for a layer that processes a single time step.

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `σ`: The non-linearity to apply to the output. Default is `tanh`.
- `init_kernel`: The initialization function to use for the input to hidden connection weights. Default is `glorot_uniform`.
- `init_recurrent_kernel`: The initialization function to use for the hidden to hidden connection weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    rnn(x, [h])

The arguments of the forward pass are:

- `x`: The input to the RNN. It should be a matrix size `in x len` or an array of size `in x len x batch_size`.
- `h`: The initial hidden state of the RNN. 
       If given, it is a vector of size `out` or a matrix of size `out x batch_size`.
       If not provided, it is assumed to be a vector of zeros, initialized by [`initialstates`](@ref).

Returns all new hidden states `h_t` as an array of size `out x len x batch_size`.

# Examples

```jldoctest
julia> d_in, d_out, len, batch_size = 4, 6, 3, 5;

julia> x = rand(Float32, (d_in, len, batch_size));

julia> h = zeros(Float32, (d_out, batch_size));

julia> rnn = RNN(d_in => d_out)
RNN(4 => 6, tanh)   # 66 parameters

julia> y = rnn(x, h);   # [y] = [d_out, len, batch_size]
```

Sometimes, the initial hidden state is a learnable parameter. 
In this case, the `RNN` should be wrapped in a custom struct.

```julia
struct Model
  rnn::RNN
  h0::AbstractVector
end

Flux.@layer Model

(m::Model)(x) = m.rnn(x, m.h0)

model = Model(RNN(32 => 64), zeros(Float32, 64))
```
"""
struct RNN{M}
  cell::M
end

@layer :noexpand RNN

initialstates(rnn::RNN) = initialstates(rnn.cell)

function RNN((in, out)::Pair, σ = tanh; cell_kwargs...)
  cell = RNNCell(in => out, σ; cell_kwargs...)
  return RNN(cell)
end

function (rnn::RNN)(x::AbstractArray)
  state = initialstates(rnn)
  return rnn(x, state)
end

function (m::RNN)(x::AbstractArray, h)
  @assert ndims(x) == 2 || ndims(x) == 3
  # [x] = [in, L] or [in, L, B]
  # [h] = [out] or [out, B]
  return scan(m.cell, x, h)
end

function Base.show(io::IO, m::RNN)
  print(io, "RNN(", size(m.cell.Wi, 2), " => ", size(m.cell.Wi, 1))
  print(io, ", ", m.cell.σ)
  print(io, ")")
end


# LSTM
@doc raw"""
    LSTMCell(in => out; init_kernel = glorot_uniform,
      init_recurrent_kernel = glorot_uniform, bias = true)

The [Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory) cell.
Behaves like an RNN but generally exhibits a longer memory span over sequences.

In the forward pass, computes

```math
i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)
f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)
c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c)
o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)
h_t = o_t \odot \tanh(c_t)
```

The `LSTMCell` returns the new hidden state `h_t` and cell state `c_t` for a single time step.
See also [`LSTM`](@ref) for a layer that processes entire sequences.

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `init_kernel`: The initialization function to use for the input to hidden connection weights. Default is `glorot_uniform`.
- `init_recurrent_kernel`: The initialization function to use for the hidden to hidden connection weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    lstmcell(x, (h, c))
    lstmcell(x)

The arguments of the forward pass are:
- `x`: The input to the LSTM. It should be a matrix of size `in` or an array of size `in x batch_size`.
- `(h, c)`: A tuple containing the hidden and cell states of the LSTM. 
  They should be vectors of size `out` or matrices of size `out x batch_size`.
  If not provided, they are assumed to be vectors of zeros, initialized by [`initialstates`](@ref).

Returns a tuple `(h′, c′)` containing the new hidden state and cell state in tensors of size  `out` or `out x batch_size`. 

# Examples

```jldoctest
julia> l = LSTMCell(3 => 5)
LSTMCell(3 => 5)    # 180 parameters

julia> h = zeros(Float32, 5); # hidden state

julia> c = zeros(Float32, 5); # cell state

julia> x = rand(Float32, 3, 4);  # in x batch_size

julia> h′, c′ = l(x, (h, c));

julia> size(h′)  # out x batch_size
(5, 4)
```
"""
struct LSTMCell{I, H, V}
  Wi::I
  Wh::H
  bias::V
end

@layer LSTMCell

function initialstates(lstm:: LSTMCell) 
  return zeros_like(lstm.Wh, size(lstm.Wh, 2)), zeros_like(lstm.Wh, size(lstm.Wh, 2))
end

function LSTMCell(
    (in, out)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true,
  )

  Wi = init_kernel(out * 4, in)
  Wh = init_recurrent_kernel(out * 4, out)
  b = create_bias(Wi, bias, out * 4)
  cell = LSTMCell(Wi, Wh, b)
  return cell
end

(lstm::LSTMCell)(x::AbstractVecOrMat) = lstm(x, initialstates(lstm))

function (m::LSTMCell)(x::AbstractVecOrMat, (h, c))
  _size_check(m, x, 1 => size(m.Wi, 2))
  b = m.bias
  g = m.Wi * x .+ m.Wh * h .+ b
  input, forget, cell, output = chunk(g, 4; dims = 1)
  c′ = @. sigmoid_fast(forget) * c + sigmoid_fast(input) * tanh_fast(cell)
  h′ = @. sigmoid_fast(output) * tanh_fast(c′)
  return h′, c′
end

Base.show(io::IO, m::LSTMCell) =
  print(io, "LSTMCell(", size(m.Wi, 2), " => ", size(m.Wi, 1) ÷ 4, ")")


@doc raw"""
    LSTM(in => out; init_kernel = glorot_uniform,
      init_recurrent_kernel = glorot_uniform, bias = true)

[Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
recurrent layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.

See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.

In the forward pass, computes

```math
i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)
f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)
c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c)
o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)
h_t = o_t \odot \tanh(c_t)
```
for all `len` steps `t` in the input sequence.
See [`LSTMCell`](@ref) for a layer that processes a single time step.

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `init_kernel`: The initialization function to use for the input to hidden connection weights. Default is `glorot_uniform`.
- `init_recurrent_kernel`: The initialization function to use for the hidden to hidden connection weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    lstm(x, (h, c))
    lstm(x)

The arguments of the forward pass are:
- `x`: The input to the LSTM. It should be a matrix of size `in x len` or an array of size `in x len x batch_size`.
- `(h, c)`: A tuple containing the hidden and cell states of the LSTM. 
    They should be vectors of size `out` or matrices of size `out x batch_size`.
    If not provided, they are assumed to be vectors of zeros, initialized by [`initialstates`](@ref).

Returns all new hidden states `h_t` as an array of size `out x len` or `out x len x batch_size`.

# Examples

```julia
struct Model
  lstm::LSTM
  h0::AbstractVector # trainable initial hidden state
  c0::AbstractVector
end

Flux.@layer Model

(m::Model)(x) = m.lstm(x, (m.h0, m.c0))

d_in, d_out, len, batch_size = 2, 3, 4, 5
x = rand(Float32, (d_in, len, batch_size))
model = Model(LSTM(d_in => d_out), zeros(Float32, d_out), zeros(Float32, d_out))
h = model(x)
size(h)  # out x len x batch_size
```
"""
struct LSTM{M}
  cell::M
end

@layer :noexpand LSTM

initialstates(lstm::LSTM) = initialstates(lstm.cell)

function LSTM((in, out)::Pair; cell_kwargs...)
  cell = LSTMCell(in => out; cell_kwargs...)
  return LSTM(cell)
end

(lstm::LSTM)(x::AbstractArray) = lstm(x, initialstates(lstm))

function (m::LSTM)(x::AbstractArray, state0)
  @assert ndims(x) == 2 || ndims(x) == 3
  return scan(m.cell, x, state0)
end

function Base.show(io::IO, m::LSTM)
  print(io, "LSTM(", size(m.cell.Wi, 2), " => ", size(m.cell.Wi, 1) ÷ 4, ")")
end

# GRU

@doc raw"""
    GRUCell(in => out; init_kernel = glorot_uniform,
      init_recurrent_kernel = glorot_uniform, bias = true)

[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078v1) layer. 
Behaves like an RNN but generally exhibits a longer memory span over sequences. 
This implements the variant proposed in v1 of the referenced paper.

In the forward pass, computes

```math
r = \sigma(W_{xi} x + W_{hi} h + b_i)
z = \sigma(W_{xz} x + W_{hz} h + b_z)
h̃ = \tanh(W_{xh} x + r \odot W_{hh} h + b_h)
h' = (1 - z) \odot h̃ + z \odot h
```

See also [`GRU`](@ref) for a layer that processes entire sequences.

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `init_kernel`: The initialization function to use for the input to hidden connection weights. Default is `glorot_uniform`.
- `init_recurrent_kernel`: The initialization function to use for the hidden to hidden connection weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    grucell(x, h)
    grucell(x)

The arguments of the forward pass are:
- `x`: The input to the GRU. It should be a vector of size `in` or a matrix of size `in x batch_size`.
- `h`: The hidden state of the GRU. It should be a vector of size `out` or a matrix of size `out x batch_size`.
  If not provided, it is assumed to be a vector of zeros, initialized by [`initialstates`](@ref).

Returns the new hidden state `h'` as an array of size `out` or `out x batch_size`.

# Examples

```jldoctest
julia> g = GRUCell(3 => 5)
GRUCell(3 => 5)     # 135 parameters

julia> h = zeros(Float32, 5); # hidden state

julia> x = rand(Float32, 3, 4);  # in x batch_size

julia> h′ = g(x, h);
```
"""
struct GRUCell{I, H, V}
  Wi::I
  Wh::H
  b::V
end

@layer GRUCell

initialstates(gru::GRUCell) = zeros_like(gru.Wh, size(gru.Wh, 2))

function GRUCell(
    (in, out)::Pair;
    init_kernel = glorot_uniform,
    init_recurrent_kernel = glorot_uniform,
    bias = true,
  )
  
  Wi = init_kernel(out * 3, in)
  Wh = init_recurrent_kernel(out * 3, out)
  b = create_bias(Wi, bias, size(Wi, 1))
  return GRUCell(Wi, Wh, b)
end

function (gru::GRUCell)(x::AbstractVecOrMat)
  state = initialstates(gru)
  return gru(x, state)
end

function (m::GRUCell)(x::AbstractVecOrMat, h)
  _size_check(m, x, 1 => size(m.Wi, 2))
  gxs = chunk(m.Wi * x, 3, dims = 1)
  ghs = chunk(m.Wh * h, 3, dims = 1)
  if m.b isa AbstractArray
    bs = chunk(m.b, 3, dims = 1)
  else # b == false
    bs = [false, false, false]
  end
  r = @. sigmoid_fast(gxs[1] + ghs[1] + bs[1])
  z = @. sigmoid_fast(gxs[2] + ghs[2] + bs[2])
  h̃ = @. tanh_fast(gxs[3] + r * ghs[3] + bs[3])
  h′ = @. (1 - z) * h̃ + z * h
  return h′
end

Base.show(io::IO, m::GRUCell) =
  print(io, "GRUCell(", size(m.Wi, 2), " => ", size(m.Wi, 1) ÷ 3, ")")

@doc raw"""
    GRU(in => out; init_kernel = glorot_uniform,
      init_recurrent_kernel = glorot_uniform, bias = true)

[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078v1) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences. This implements
the variant proposed in v1 of the referenced paper.

The forward pass computes

```math
r_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)
z_t = \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z)
h̃_t = \tanh(W_{xh} x_t + r_t \odot W_{hh} h_{t-1} + b_h)
h_t = (1 - z_t) \odot h̃_t + z_t \odot h_{t-1}
```
for all `len` steps `t` in the input sequence.
See [`GRUCell`](@ref) for a layer that processes a single time step.

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `init_kernel`: The initialization function to use for the input to hidden connection weights. Default is `glorot_uniform`.
- `init_recurrent_kernel`: The initialization function to use for the hidden to hidden connection weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    gru(x, [h])

The arguments of the forward pass are:

- `x`: The input to the GRU. It should be a matrix of size `in x len` or an array of size `in x len x batch_size`.
- `h`: The initial hidden state of the GRU. It should be a vector of size `out` or a matrix of size `out x batch_size`.
       If not provided, it is assumed to be a vector of zeros, initialized by [`initialstates`](@ref).

Returns all new hidden states `h_t` as an array of size `out x len x batch_size`.

# Examples

```julia
d_in, d_out, len, batch_size = 2, 3, 4, 5
gru = GRU(d_in => d_out)
x = rand(Float32, (d_in, len, batch_size))
h0 = zeros(Float32, d_out)
h = gru(x, h0)  # out x len x batch_size
```
"""
struct GRU{M}
  cell::M
end

@layer :noexpand GRU

initialstates(gru::GRU) = initialstates(gru.cell)

function GRU((in, out)::Pair; cell_kwargs...)
  cell = GRUCell(in => out; cell_kwargs...)
  return GRU(cell)
end

(gru::GRU)(x::AbstractArray) = gru(x, initialstates(gru))

function (m::GRU)(x::AbstractArray, h)
  @assert ndims(x) == 2 || ndims(x) == 3
  return scan(m.cell, x, h)
end

function Base.show(io::IO, m::GRU)
  print(io, "GRU(", size(m.cell.Wi, 2), " => ", size(m.cell.Wi, 1) ÷ 3, ")")
end

# GRU v3
@doc raw"""
    GRUv3Cell(in => out; init_kernel = glorot_uniform,
      init_recurrent_kernel = glorot_uniform, bias = true)
    
[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078v3) layer. 
Behaves like an RNN but generally exhibits a longer memory span over sequences. 
This implements the variant proposed in v3 of the referenced paper.

The forward pass computes
```math
r = \sigma(W_{xi} x + W_{hi} h + b_i)
z = \sigma(W_{xz} x + W_{hz} h + b_z)
h̃ = \tanh(W_{xh} x + W_{hh̃} (r \odot W_{hh} h) + b_h)
h' = (1 - z) \odot h̃ + z \odot h
```
and returns `h'`. This is a single time step of the GRU.

See [`GRUv3`](@ref) for a layer that processes entire sequences.
See [`GRU`](@ref) and [`GRUCell`](@ref) for variants of this layer.

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `init_kernel`: The initialization function to use for the input to hidden connection weights. Default is `glorot_uniform`.
- `init_recurrent_kernel`: The initialization function to use for the hidden to hidden connection weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    gruv3cell(x, [h])

The arguments of the forward pass are:
- `x`: The input to the GRU. It should be a vector of size `in` or a matrix of size `in x batch_size`.
- `h`: The hidden state of the GRU. It should be a vector of size `out` or a matrix of size `out x batch_size`.
  If not provided, it is assumed to be a vector of zeros, initialized by [`initialstates`](@ref).

Returns the new hidden state `h'` as an array of size `out` or `out x batch_size`.
"""
struct GRUv3Cell{I, H, V, HH}
  Wi::I
  Wh::H
  b::V
  Wh_h̃::HH
end

@layer GRUv3Cell

initialstates(gru::GRUv3Cell) = zeros_like(gru.Wh, size(gru.Wh, 2))

function GRUv3Cell(
  (in, out)::Pair;
  init_kernel = glorot_uniform,
  init_recurrent_kernel = glorot_uniform,
  bias = true,
)
  Wi = init_kernel(out * 3, in)
  Wh = init_recurrent_kernel(out * 3, out)
  Wh_h̃ = init_recurrent_kernel(out, out)
  b = create_bias(Wi, bias, out * 3)
  return GRUv3Cell(Wi, Wh, b, Wh_h̃)
end

function (gru::GRUv3Cell)(x::AbstractVecOrMat)
  state = initialstates(gru)
  return gru(x, state)
end

function (m::GRUv3Cell)(x::AbstractVecOrMat, h)
  _size_check(m, x, 1 => size(m.Wi, 2))
  gxs = chunk(m.Wi * x, 3, dims = 1)
  ghs = chunk(m.Wh * h, 3, dims = 1)
  if m.b isa AbstractArray
    bs = chunk(m.b, 3, dims = 1)
  else # m.b == false
    bs = [false, false, false]
  end
  r = @. sigmoid_fast(gxs[1] + ghs[1] + bs[1])
  z = @. sigmoid_fast(gxs[2] + ghs[2] + bs[2])
  h̃ = tanh_fast.(gxs[3] .+ (m.Wh_h̃ * (r .* h)) .+ bs[3])
  h′ = @. (1 - z) * h̃ + z * h
  return h′
end

Base.show(io::IO, m::GRUv3Cell) =
  print(io, "GRUv3Cell(", size(m.Wi, 2), " => ", size(m.Wi, 1) ÷ 3, ")")


@doc raw"""
    GRUv3(in => out; init_kernel = glorot_uniform,
      init_recurrent_kernel = glorot_uniform, bias = true)

[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078v3) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences. This implements
the variant proposed in v3 of the referenced paper.

The forward pass computes

```math
r_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)
z_t = \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z)
h̃_t = \tanh(W_{xh} x_t + W_{hh̃} (r_t \odot  W_{hh} h_{t-1}) + b_h)
h_t = (1 - z_t) \odot h̃_t + z_t \odot h_{t-1}
```
for all `len` steps `t` in the input sequence. 
See [`GRUv3Cell`](@ref) for a layer that processes a single time step.
See [`GRU`](@ref) and [`GRUCell`](@ref) for variants of this layer.

Notice that `GRUv3` is not a more advanced version of [`GRU`](@ref)
but only a less popular variant.

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `init_kernel`: The initialization function to use for the input to hidden connection weights. Default is `glorot_uniform`.
- `init_recurrent_kernel`: The initialization function to use for the hidden to hidden connection weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    gruv3(x, [h])

The arguments of the forward pass are:

- `x`: The input to the GRU. It should be a matrix of size `in x len` or an array of size `in x len x batch_size`.
- `h`: The initial hidden state of the GRU. It should be a vector of size `out` or a matrix of size `out x batch_size`.
       If not provided, it is assumed to be a vector of zeros, initialized by [`initialstates`](@ref).

Returns all new hidden states `h_t` as an array of size `out x len x batch_size`.

# Examples

```julia
d_in, d_out, len, batch_size = 2, 3, 4, 5
gruv3 = GRUv3(d_in => d_out)
x = rand(Float32, (d_in, len, batch_size))
h0 = zeros(Float32, d_out)
h = gruv3(x, h0)  # out x len x batch_size
```
"""
struct GRUv3{M}
  cell::M
end

@layer :noexpand GRUv3

initialstates(gru::GRUv3) = initialstates(gru.cell)

function GRUv3((in, out)::Pair; cell_kwargs...)
  cell = GRUv3Cell(in => out; cell_kwargs...)
  return GRUv3(cell)
end

(gru::GRUv3)(x::AbstractArray) = gru(x, initialstates(gru))

function (m::GRUv3)(x::AbstractArray, h)
  @assert ndims(x) == 2 || ndims(x) == 3
  return scan(m.cell, x, h)
end

function Base.show(io::IO, m::GRUv3)
  print(io, "GRUv3(", size(m.cell.Wi, 2), " => ", size(m.cell.Wi, 1) ÷ 3, ")")
end