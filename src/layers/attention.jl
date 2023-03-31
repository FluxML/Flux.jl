
const A3{T} = AbstractArray{T, 3}
const IntOrDims{N} = Union{Int, Dims{N}}

"""
    MultiHeadAttention(dims; [nheads, bias, init, dropout_prob])

The multi-head dot-product attention layer used in Transformer architectures [1].

Returns the transformed input sequnce and the attention scores.

[1] Vaswani et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.

# Arguments

- `dims`: The embedding dimensions of inputs, intermediate tensors and outputs.
          In the most general case, it is given as 
          a) `(q_in_dim, k_in_dim, v_in_dim) => (qk_dim, v_dim) => out_dim`.
          Can take also simpler forms as
          b) `dims::Int`;
          c) `in_dim::Int => (qk_dim, v_dim) => out_dim`;
          d) `in_dim::Int => qkv_dim => out_dim`.
- `nheads`: number of heads. Default `8`.
- `init`: weight initializer for the Dense layers. Default `glorot_uniform`.
- `bias` : whether pointwise QKVO dense transforms use bias. Default `false`.
- `dropout_prob`: dropout probability for the attention scores. Default `0.0`.

# Forward
    
    (mha::MultiHeadAttention)(q_in, k_in, v_in, [bias]; [mask])

The arguments of the forward pass are:

- `q_in`: Input query array of size `(q_in_dim, q_len, batch_size)`.
- `k_in`: Input key array of size `(k_in_dim, kv_len, batch_size)`.
- `v_in`: Input value array of size `(v_in_dim, kv_len, batch_size)`.
- `bias`: Bias array broadcastable to size `(kv_len, q_len, nheads, batch_size)`. 
          It will be added to the attention scores before the softmax.
          Default `nothing`.
- `mask`: Input array broadcastable to size 
          `(kv_len, q_len, nheads, batch_size)`. 
          The mask is applied to the attention scores just before the softmax. 
          See [`NNlib.make_causal_mask`](@ref) for creating causal masks. 
          Default `nothing`.

Alternative calling signatures are `mha(q_in)`, equivalent to `mha(q_in, q_in, q_in)` (self-attention),
and `mha(q_in, k_in)`, equivalent to `mha(q_in, k_in, k_in)` (key and value are the same).

See also [`NNlib.dot_product_attention`](@ref).

# Examples

```julia
mha = MultiHeadAttention(64, nheads = 8)
q = rand(Float32, (64, 10, 32))
k = rand(Float32, (64, 20, 32))
v = rand(Float32, (64, 20, 32))
y, α = mha(q, k, v) 
# [y] = [64, 10, 32]
# [α] = [20, 10, 8, 32]

mha = MultiHeadAttention(64 => 1024 => 1024, nheads = 8)
y, α = mha(q) # self-attention
# [y] = [1024, 10, 32]
# [α] = [10, 10, 8, 32]
```
"""
struct MultiHeadAttention{P1, D, P2}
  nheads::Int
  q_proj::P1
  k_proj::P1
  v_proj::P1
  attn_drop::D
  out_proj::P2
end

@functor MultiHeadAttention

function MultiHeadAttention(dims; 
                     nheads::Int = 8,
                     bias::Bool = false,
                     init = glorot_uniform,                    
                     dropout_prob = 0.0)

  dims = normalize_mha_dims(dims)
  @assert dims.qk % nheads == 0 "qk_dim should be divisible by nheads"
  @assert dims.v % nheads == 0 "v_dim should be divisible by nheads"
  q_proj = Dense(dims.q_in => dims.qk; bias, init)
  k_proj = Dense(dims.k_in => dims.qk; bias, init)
  v_proj = Dense(dims.v_in => dims.v; bias, init)
  attn_drop = Dropout(dropout_prob)
  out_proj = Dense(dims.v => dims.out; bias, init)
  return MultiHeadAttention(nheads, q_proj, k_proj, v_proj, attn_drop, out_proj)
end

# turns the dims argument into a named tuple
normalize_mha_dims(dims::Int) = 
  (; q_in=dims, k_in=dims, v_in=dims, qk=dims, v=dims, out=dims)

function normalize_mha_dims((in, (qkv, out))::Pair{<:IntOrDims{3}, <:Pair{<:IntOrDims{2}, Int}})
  if in isa Int
    q_in = k_in = v_in = in
  else
    q_in, k_in, v_in = in
  end
  if qkv isa Int
    qk = v = qkv
  else
    qk, v = qkv
  end
  return (; q_in, k_in, v_in, qk, v, out)
end

# self-attention
(mha::MultiHeadAttention)(qkv; kws...) = mha(qkv, qkv, qkv; kws...)

# key and value are the same
(mha::MultiHeadAttention)(q, kv; kws...) = mha(q, kv, kv; kws...)

function (mha::MultiHeadAttention)(q_in::A3, k_in::A3, v_in::A3, 
                                  bias=nothing; mask=nothing)
  ## [q_in] = [q_in_dim, q_len, batch_size]
  ## [k_in] = [k_in_dim, kv_len, batch_size] 
  ## [v_in] = [v_in_dim, kv_len, batch_size]
  q = mha.q_proj(q_in)  # [q] = [qk_dim, q_len, batch_size]
  k = mha.k_proj(k_in)  # [k] = [qk_dim, kv_len, batch_size] 
  v = mha.v_proj(v_in)  # [v] = [v_dim, kv_len, batch_size]
  x, α = NNlib.dot_product_attention(q, k, v, bias; mha.nheads, mask, fdrop=mha.attn_drop)
  x = mha.out_proj(x)
  # [x] = [out_dim, q_len, batch_size]
  # [α] = [kv_len, q_len, nheads, batch_size]
  return x, α
end
