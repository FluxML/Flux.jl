
const A3{T} = AbstractArray{T, 3}
const IntOrDims{N} = Union{Int, Dims{N}}

"""
    Attention(dims; [nheads, nkvheads, bias, init, dropout, qk_norm, is_causal])

The multi-head scaled dot-product attention layer used in Transformer architectures [1].

Returns the transformed input sequence. The attention weights are not computed by the
forward pass, see [`Flux.attention_with_weights`](@ref) to obtain them.

[1] Vaswani et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.

# Arguments

- `dims`: The embedding dimensions of inputs, intermediate tensors and outputs.
          In the most general case, it is given as
          a) `(q_in_dim, k_in_dim, v_in_dim) => (qk_dim, v_dim) => out_dim`.
          Can take also simpler forms as
          b) `dims::Int`;
          c) `in_dim::Int => (qk_dim, v_dim) => out_dim`;
          d) `in_dim::Int => qkv_dim => out_dim`.
- `nheads`: number of heads. Each head has dimension `qk_dim ÷ nheads`. Default `8`.
- `nkvheads`: number of key and value heads. If smaller than `nheads`, each of them is shared
          by `nheads ÷ nkvheads` query heads (grouped-query attention). Default `nheads`.
- `bias` : whether pointwise QKVO dense transforms use bias. Default `false`.
- `init`: weight initializer for the Dense layers. Default `glorot_uniform`.
- `dropout`: dropout probability for the attention weights. Default `0.0`.
- `qk_norm`: either `nothing` or a function `head_dim -> layer` constructing the normalization
          applied to the queries and to the keys of each head, e.g. `qk_norm = LayerNorm`.
          Default `nothing`.
- `is_causal`: if `true`, each position attends only to itself and to the preceding ones.
          Default `false`.

# Forward

    (m::Attention)(q_in, k_in = q_in, v_in = k_in; [mask, bias, is_causal, q_lengths, kv_lengths])

The arguments of the forward pass are:

- `q_in`: Input query array of size `(q_in_dim, q_len, batch_size...)`.
- `k_in`: Input key array of size `(k_in_dim, kv_len, batch_size...)`. Default `q_in` (self-attention).
- `v_in`: Input value array of size `(v_in_dim, kv_len, batch_size...)`. Default `k_in`.
- `mask`: Boolean array broadcastable to size `(kv_len, q_len, nheads, batch_size...)`,
          with `false` marking the entries that cannot be attended.
          A query attending to no key at all produces `NaN`s. Default `nothing`.
- `bias`: Array broadcastable to size `(kv_len, q_len, nheads, batch_size...)`,
          added to the attention logits before the softmax. Default `nothing`.
- `is_causal`: overrides the `is_causal` setting of the layer.
- `q_lengths`, `kv_lengths`: the number of valid positions of each query and of each key/value
          sequence in the batch, the following ones being padding: an integer array with one
          entry for each batch element, or an integer for all of them. The keys past `kv_lengths`
          are not attended to, and the attention output is zero at the queries past `q_lengths`
          (before the output projection). Default `nothing`.

`mask`, `is_causal` and `kv_lengths` can be combined: a key is then attended to only where all
of them allow it.

The output has size `(out_dim, q_len, batch_size...)`.

# Performance

Unless the attention dropout is active, the layer takes a fused implementation of
[`NNlib.scaled_dot_product_attention`](@ref), which does not materialize the attention weights,
so that its memory grows linearly instead of quadratically with the sequence length. This is the
case on the CPU and on GPUs, and for the gradient with Zygote and Enzyme as long as there is no
`bias`; see the NNlib docstring for the details.

With CUDA.jl and cuDNN.jl loaded, a layer converted with [`f16`](@ref Flux.f16) or
[`bf16`](@ref Flux.bf16) uses instead cuDNN's fused kernel, which runs on tensor cores.
This requires no `mask` and no `bias`, a head dimension that is a multiple of 8 and at most 256,
and `qk_dim == v_dim`. A causal layer should therefore use `is_causal = true` rather than a
causal `mask`, and a padded batch `kv_lengths` rather than a padding `mask`.

This layer covers the standard attention block. Architectures that transform the queries and
keys in other ways (e.g. rotary embeddings) or that cache keys and values can be built on
[`NNlib.scaled_dot_product_attention`](@ref), which this layer wraps.

# Examples

```julia
attn = Attention(64, nheads = 8)
q = rand(Float32, (64, 10, 32))
k = rand(Float32, (64, 20, 32))
v = rand(Float32, (64, 20, 32))
y = attn(q, k, v)
# [y] = [64, 10, 32]

# a padded batch, whose first key/value sequence has 12 valid positions out of 20
y = attn(q, k, v; kv_lengths = [12; fill(20, 31)])

attn = Attention(64 => 1024 => 1024, nheads = 8, is_causal = true)
y = attn(q) # causal self-attention
# [y] = [1024, 10, 32]

y, α = Flux.attention_with_weights(attn, q)
# [α] = [10, 10, 8, 32]
```
"""
struct Attention{Q, K, V, QN, KN, D, O}
  nheads::Int
  nkvheads::Int
  q_proj::Q
  k_proj::K
  v_proj::V
  q_norm::QN
  k_norm::KN
  attn_drop::D
  out_proj::O
  is_causal::Bool
end

@layer :noexpand Attention

function Attention(dims;
                   nheads::Int = 8,
                   nkvheads::Int = nheads,
                   bias::Bool = false,
                   init = glorot_uniform,
                   dropout::Real = 0.0,
                   qk_norm = nothing,
                   is_causal::Bool = false)

  dims = normalize_mha_dims(dims)
  dims.qk % nheads == 0 || throw(ArgumentError("qk_dim = $(dims.qk) should be divisible by nheads = $(nheads)"))
  dims.v % nheads == 0 || throw(ArgumentError( "v_dim = $(dims.v) should be divisible by nheads = $(nheads)"))
  nheads % nkvheads == 0 || throw(ArgumentError("nheads = $(nheads) should be divisible by nkvheads = $(nkvheads)"))
  qk_head_dim = dims.qk ÷ nheads
  v_head_dim = dims.v ÷ nheads
  q_proj = Dense(dims.q_in => dims.qk; bias, init)
  k_proj = Dense(dims.k_in => qk_head_dim * nkvheads; bias, init)
  v_proj = Dense(dims.v_in => v_head_dim * nkvheads; bias, init)
  q_norm = qk_norm === nothing ? identity : qk_norm(qk_head_dim)
  k_norm = qk_norm === nothing ? identity : qk_norm(qk_head_dim)
  attn_drop = Dropout(dropout)
  out_proj = Dense(dims.v => dims.out; bias, init)
  return Attention(nheads, nkvheads, q_proj, k_proj, v_proj, q_norm, k_norm, attn_drop, out_proj, is_causal)
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

function (m::Attention)(q_in::AbstractArray, k_in::AbstractArray = q_in, v_in::AbstractArray = k_in;
                        mask = nothing, bias = nothing, is_causal::Bool = m.is_causal,
                        q_lengths = nothing, kv_lengths = nothing)
  ## [q_in] = [q_in_dim, q_len, batch_size...]
  ## [k_in] = [k_in_dim, kv_len, batch_size...]
  ## [v_in] = [v_in_dim, kv_len, batch_size...]
  q, k, v = _project_qkv(m, q_in, k_in, v_in)
  # Only without `fdrop` can NNlib take a fused implementation.
  x = if _dropout_active(m.attn_drop, q)
    NNlib.scaled_dot_product_attention(q, k, v, bias; mask, is_causal, q_lengths, kv_lengths,
                                       fdrop = m.attn_drop)
  else
    NNlib.scaled_dot_product_attention(q, k, v, bias; mask, is_causal, q_lengths, kv_lengths)
  end
  # [x] = [v_head_dim, nheads, q_len, batch_size...]
  return m.out_proj(_merge_heads(x))
end

"""
    attention_with_weights(m::Attention, q_in, k_in = q_in, v_in = k_in; [mask, bias, is_causal, q_lengths, kv_lengths])

Return the output `y` of the [`Attention`](@ref) layer `m` together with the attention weights `α`
it is computed from. The arguments are those of the forward pass of `m`, and `y` is the same as
`m(q_in, k_in, v_in; kws...)`.

The weights have size `(kv_len, q_len, nheads, batch_size...)`, with `α[j, i, h, b...]` the weight
of the key `j` for the query `i` in the head `h`. They are taken after the attention dropout if
it is active, and are zero at the queries past `q_lengths`.

Returning the weights means materializing them, so this function never takes the fused
implementation of the forward pass.

# Examples

```julia
attn = Attention(64, nheads = 8)
q = rand(Float32, (64, 10, 32))
k = rand(Float32, (64, 20, 32))
y, α = Flux.attention_with_weights(attn, q, k)
# [y] = [64, 10, 32]
# [α] = [20, 10, 8, 32]
```
"""
function attention_with_weights(m::Attention, q_in::AbstractArray, k_in::AbstractArray = q_in,
                                v_in::AbstractArray = k_in;
                                mask = nothing, bias = nothing, is_causal::Bool = m.is_causal,
                                q_lengths = nothing, kv_lengths = nothing)
  q, k, v = _project_qkv(m, q_in, k_in, v_in)
  x, α = NNlib.scaled_dot_product_attention_with_weights(q, k, v, bias; mask, is_causal,
                                                         q_lengths, kv_lengths, fdrop = m.attn_drop)
  return m.out_proj(_merge_heads(x)), α
end

function _project_qkv(m::Attention, q_in, k_in, v_in)
  q = m.q_norm(_split_heads(m.q_proj(q_in), m.nheads))    # [q] = [qk_head_dim, nheads, q_len, batch_size...]
  k = m.k_norm(_split_heads(m.k_proj(k_in), m.nkvheads))  # [k] = [qk_head_dim, nkvheads, kv_len, batch_size...]
  v = _split_heads(m.v_proj(v_in), m.nkvheads)            # [v] = [v_head_dim, nkvheads, kv_len, batch_size...]
  return q, k, v
end

_split_heads(x, nheads) = reshape(x, :, nheads, size(x)[2:end]...)
_merge_heads(x) = reshape(x, :, size(x)[3:end]...)

# This has to be differentiated through: `within_gradient` is true only inside its rrule.
_dropout_active(d::Dropout, x) = d.p > 0 && _isactive(d, x)

function Base.show(io::IO, m::Attention)
  qk, q_in = size(m.q_proj.weight)
  k_in = size(m.k_proj.weight, 2)
  v_in = size(m.v_proj.weight, 2)
  out, v = size(m.out_proj.weight)
  print(io, "Attention(")
  if q_in == k_in == v_in == qk == v == out
    print(io, q_in)
  elseif q_in == k_in == v_in && qk == v
    print(io, q_in, " => ", qk, " => ", out)
  elseif q_in == k_in == v_in
    print(io, q_in, " => (", qk, ", ", v,") => ", out)
  else
    print(io, "(", q_in, ", ", k_in, ", ", v_in, ") => (", qk, ", ", v,") => ", out)
  end
  print(io, "; nheads=", m.nheads)
  m.nkvheads == m.nheads || print(io, ", nkvheads=", m.nkvheads)
  m.q_proj.bias === false || print(io, ", bias=true")
  m.attn_drop.p == 0 || print(io, ", dropout=", m.attn_drop.p)
  m.q_norm === identity || print(io, ", qk_norm=", nameof(typeof(m.q_norm)))
  m.is_causal && print(io, ", is_causal=true")
  print(io, ")")
end
