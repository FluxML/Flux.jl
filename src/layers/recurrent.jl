
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

See [`RNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `σ`: The non-linearity to apply to the output. Default is `tanh`.
- `init`: The initialization function to use for the weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    rnncell(x, [h])

The arguments of the forward pass are:

- `x`: The input to the RNN. It should be a vector of size `in` or a matrix of size `in x batch_size`.
- `h`: The hidden state of the RNN. It should be a vector of size `out` or a matrix of size `out x batch_size`.
       If not provided, it is assumed to be a vector of zeros.

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

(m::RNNCell)(x::AbstractVecOrMat) = m(x, zeros_like(x, size(m.Wh, 1)))

function (m::RNNCell)(x::AbstractVecOrMat, h::AbstractVecOrMat)
  _size_check(m, x, 1 => size(m.Wi,2))
  σ = NNlib.fast_act(m.σ, x)
  h = σ.(m.Wi*x .+ m.Wh*h .+ m.bias)
  return h
end

function Base.show(io::IO, m::RNNCell)
  print(io, "RNNCell(", size(m.Wi, 2), " => ", size(m.Wi, 1))
  print(io, ", ", m.σ)
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

See [`RNNCell`](@ref) for a layer that processes a single time step.

# Arguments

- `in => out`: The input and output dimensions of the layer.
- `σ`: The non-linearity to apply to the output. Default is `tanh`.
- `init`: The initialization function to use for the weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    rnn(x, h)

The arguments of the forward pass are:

- `x`: The input to the RNN. It should be a matrix size `in x len` or an array of size `in x len x batch_size`.
- `h`: The initial hidden state of the RNN. It should be a vector of size `out` or a matrix of size `out x batch_size`.

Returns all new hidden states `h_t` as an array of size `out x len x batch_size`.

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

(m::RNN)(x) = m(x, zeros_like(x, size(m.cell.Wh, 1)))

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
@doc raw"""
    LSTMCell(in => out; init = glorot_uniform, bias = true)

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
- `init`: The initialization function to use for the weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    lstmcell(x, (h, c))
    lstmcell(x)

The arguments of the forward pass are:
- `x`: The input to the LSTM. It should be a matrix of size `in` or an array of size `in x batch_size`.
- `(h, c)`: A tuple containing the hidden and cell states of the LSTM. 
  They should be vectors of size `out` or matrices of size `out x batch_size`.
  If not provided, they are assumed to be vectors of zeros.

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
"""
struct LSTMCell{I,H,V}
  Wi::I
  Wh::H
  bias::V
end

@layer LSTMCell

function LSTMCell((in, out)::Pair; init = glorot_uniform, bias = true)
  Wi = init(out * 4, in)
  Wh = init(out * 4, out)
  b = create_bias(Wi, bias, out * 4)
  cell = LSTMCell(Wi, Wh, b)
  return cell
end

function (m::LSTMCell)(x::AbstractVecOrMat)
  h = zeros_like(x, size(m.Wh, 2))
  c = zeros_like(h)
  return m(x, (h, c))
end

function (m::LSTMCell)(x::AbstractVecOrMat, (h, c))
  _size_check(m, x, 1 => size(m.Wi, 2))
  b = m.bias
  g = m.Wi * x .+ m.Wh * h .+ b
  input, forget, cell, output = chunk(g, 4; dims=1)
  c′ = @. sigmoid_fast(forget) * c + sigmoid_fast(input) * tanh_fast(cell)
  h′ = @. sigmoid_fast(output) * tanh_fast(c′)
  return h′, c′
end

Base.show(io::IO, m::LSTMCell) =
  print(io, "LSTMCell(", size(m.Wi, 2), " => ", size(m.Wi, 1)÷4, ")")


@doc raw""""
    LSTM(in => out; init = glorot_uniform, bias = true)

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
- `init`: The initialization function to use for the weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    lstm(x, (h, c))
    lstm(x)

The arguments of the forward pass are:
- `x`: The input to the LSTM. It should be a matrix of size `in x len` or an array of size `in x len x batch_size`.
- `(h, c)`: A tuple containing the hidden and cell states of the LSTM. 
    They should be vectors of size `out` or matrices of size `out x batch_size`.
    If not provided, they are assumed to be vectors of zeros.

Returns a tuple `(h′, c′)` containing all new hidden states `h_t` and cell states `c_t` 
in tensors of size `out x len` or `out x len x batch_size`.

# Examples

```jldoctest
struct Model
  lstm::LSTM
  h0::AbstractVector
  c0::AbstractVector
end

Flux.@layer :expand Model

(m::Model)(x) = m.lstm(x, (m.h0, m.c0))

d_in, d_out, len, batch_size = 2, 3, 4, 5
x = rand(Float32, (d_in, len, batch_size))
model = Model(LSTM(d_in => d_out), zeros(Float32, d_out), zeros(Float32, d_out))
h, c = model(x)
size(h)  # out x len x batch_size
```
"""
struct LSTM{M}
  cell::M
end

@layer :expand LSTM

function LSTM((in, out)::Pair; init = glorot_uniform, bias = true)
  cell = LSTMCell(in => out; init, bias)
  return LSTM(cell)
end

function (m::LSTM)(x)
  h = zeros_like(x, size(m.cell.Wh, 1))
  c = zeros_like(h)
  return m(x, (h, c))
end

function (m::LSTM)(x, (h, c))
  @assert ndims(x) == 2 || ndims(x) == 3
  h′ = []
  c′ = []
  for x_t in eachslice(x, dims=2)
    h, c = m.cell(x_t, (h, c))
    h′ = vcat(h′, [h])
    c′ = vcat(c′, [c])
  end
  return stack(h′, dims=2), stack(c′, dims=2)
end

# GRU

@doc raw"""
    GRUCell(in => out; init = glorot_uniform, bias = true)

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
- `init`: The initialization function to use for the weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    grucell(x, h)
    grucell(x)

The arguments of the forward pass are:
- `x`: The input to the GRU. It should be a vector of size `in` or a matrix of size `in x batch_size`.
- `h`: The hidden state of the GRU. It should be a vector of size `out` or a matrix of size `out x batch_size`.
  If not provided, it is assumed to be a vector of zeros.

Returns the new hidden state `h'` as an array of size `out` or `out x batch_size`.

# Examples

TODO add loop
```jldoctest
julia> g = GRUCell(3 => 5)
GRUCell(3 => 5)    # 140 parameters

julia> h = zeros(Float32, 5); # hidden state

julia> x = rand(Float32, 3, 4);  # in x batch_size

julia> h′ = g(x, h);
```
"""
struct GRUCell{I,H,V}
  Wi::I
  Wh::H
  b::V
end

@layer GRUCell

function GRUCell((in, out)::Pair; init = glorot_uniform, bias = true)
  Wi = init(out * 3, in)
  Wh = init(out * 3, out)
  b = create_bias(Wi, bias, size(Wi, 1))
  return GRUCell(Wi, Wh, b)
end

(m::GRUCell)(x::AbstractVecOrMat) = m(x, zeros_like(x, size(m.Wh, 2)))

function (m::GRUCell)(x::AbstractVecOrMat, h)
  _size_check(m, x, 1 => size(m.Wi,2))
  gxs = chunk(m.Wi * x, 3, dims=1)
  ghs = chunk(m.Wh * h, 3, dims=1)
  if m.b isa AbstractArray
    bs = chunk(m.b, 3, dims=1)
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
  print(io, "GRUCell(", size(m.Wi, 2), " => ", size(m.Wi, 1)÷3, ")")

@doc raw"""
    GRU(in => out; init = glorot_uniform, bias = true)

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

# Forward

    gru(x, h)
    gru(x)

The arguments of the forward pass are:

- `x`: The input to the GRU. It should be a matrix of size `in x len` or an array of size `in x len x batch_size`.
- `h`: The initial hidden state of the GRU. It should be a vector of size `out` or a matrix of size `out x batch_size`.
       If not provided, it is assumed to be a vector of zeros. 

Returns all new hidden states `h_t` as an array of size `out x len x batch_size`.

# Examples

```jldoctest
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

@layer :expand GRU

function GRU((in, out)::Pair; init = glorot_uniform, bias = true)
  cell = GRUCell(in => out; init, bias)
  return GRU(cell)
end

function (m::GRU)(x)
  h = zeros_like(x, size(m.cell.Wh, 2))
  return m(x, h)
end

function (m::GRU)(x, h)
  @assert ndims(x) == 2 || ndims(x) == 3
  h′ = []
  # [x] = [in, L] or [in, L, B]
  for x_t in eachslice(x, dims=2)
    h = m.cell(x_t, h)
    h′ = vcat(h′, [h])
  end
  return stack(h′, dims=2)
end

# GRU v3
@doc raw"""
    GRUv3Cell(in => out, init = glorot_uniform, bias = true)
    
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
- `init`: The initialization function to use for the weights. Default is `glorot_uniform`.
- `bias`: Whether to include a bias term initialized to zero. Default is `true`.

# Forward

    gruv3cell(x, h)
    gruv3cell(x)

The arguments of the forward pass are:
- `x`: The input to the GRU. It should be a vector of size `in` or a matrix of size `in x batch_size`.
- `h`: The hidden state of the GRU. It should be a vector of size `out` or a matrix of size `out x batch_size`.
  If not provided, it is assumed to be a vector of zeros.

Returns the new hidden state `h'` as an array of size `out` or `out x batch_size`.
"""
struct GRUv3Cell{I,H,V,HH}
  Wi::I
  Wh::H
  b::V
  Wh_h̃::HH
end

@layer GRUv3Cell

function GRUv3Cell((in, out)::Pair; init = glorot_uniform, bias = true)
  Wi = init(out * 3, in)
  Wh = init(out * 3, out)
  Wh_h̃ = init(out, out)
  b = create_bias(Wi, bias, out * 3)
  return GRUv3Cell(Wi, Wh, b, Wh_h̃)
end

(m::GRUv3Cell)(x::AbstractVecOrMat) = m(x, zeros_like(x, size(m.Wh, 2)))

function (m::GRUv3Cell)(x::AbstractVecOrMat, h)
  _size_check(m, x, 1 => size(m.Wi,2))
  gxs = chunk(m.Wi * x, 3, dims=1)
  ghs = chunk(m.Wh * h, 3, dims=1)
  if m.b isa AbstractArray
    bs = chunk(m.b, 3, dims=1)
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
  print(io, "GRUv3Cell(", size(m.Wi, 2), " => ", size(m.Wi, 1)÷3, ")")


@doc raw"""
    GRUv3(in => out)

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

# Examples
TODO
"""
struct GRUv3{M}
  cell::M
end

@layer :expand GRUv3

function GRUv3((in, out)::Pair; init = glorot_uniform, bias = true)
  cell = GRUv3Cell(in => out; init, bias)
  return GRUv3(cell)
end

function (m::GRUv3)(x)
  h = zeros_like(x, size(m.cell.Wh, 2))
  return m(x, h)
end

function (m::GRUv3)(x, h)
  @assert ndims(x) == 2 || ndims(x) == 3
  h′ = []
  for x_t in eachslice(x, dims=2)
    h = m.cell(x_t, h)
    h′ = vcat(h′, [h])
  end
  return stack(h′, dims=2)
end

