"""
    MHAttention(planes::Integer, nheads::Integer = 8; qkv_bias::Bool = false, 
                attn_dropout_prob = 0., proj_dropout_prob = 0.)

Multi-head self-attention layer.

# Arguments

- `planes`: number of input channels
- `nheads`: number of heads
- `qkv_bias`: whether to use bias in the layer to get the query, key and value
- `attn_dropout_prob`: dropout probability after the self-attention layer
- `proj_dropout_prob`: dropout probability after the projection layer
"""
struct MultiHeadAttention{P, Q, R}
  nheads::Int
  qkv_layer::P
  attn_drop::Q
  projection::R
end

@functor MHAttention

function MultiHeadAttention(planes::Integer, nheads::Integer = 8; qkv_bias::Bool = false,
                     attn_dropout_prob = 0.0, proj_dropout_prob = 0.0)
    @assert planes % nheads==0 "planes should be divisible by nheads"
    qkv_layer = Dense(planes, planes * 3; bias = qkv_bias)
    attn_drop = Dropout(attn_dropout_prob)
    proj = Chain(Dense(planes, planes), Dropout(proj_dropout_prob))
    return MultiHeadAttention(nheads, qkv_layer, attn_drop, proj)
end

function (m::MultiHeadAttention)(x::AbstractArray{T, 3}) where {T}
    nfeatures, seq_len, batch_size = size(x)
    x_reshaped = reshape(x, nfeatures, seq_len * batch_size)
    qkv = m.qkv_layer(x_reshaped)
    qkv_reshaped = reshape(qkv, nfeatures ÷ m.nheads, m.nheads, seq_len, 3 * batch_size)
    query, key, value = chunk(qkv_reshaped, 3; dims = 4)
    scale = convert(T, sqrt(size(query, 1) / m.nheads))
    key_reshaped = reshape(permutedims(key, (2, 1, 3, 4)), m.nheads, nfeatures ÷ m.nheads,
                           seq_len * batch_size)
    query_reshaped = reshape(permutedims(query, (1, 2, 3, 4)), nfeatures ÷ m.nheads,
                             m.nheads, seq_len * batch_size)

    attention = softmax(batched_mul(query_reshaped, key_reshaped) .* scale)                         
    attention = m.attn_drop(attention)
    
    value_reshaped = reshape(permutedims(value, (1, 2, 3, 4)), nfeatures ÷ m.nheads,
                             m.nheads, seq_len * batch_size)
    pre_projection = reshape(batched_mul(attention, value_reshaped),
                             (nfeatures, seq_len, batch_size))
    y = m.projection(reshape(pre_projection, size(pre_projection, 1), :))
    return reshape(y, :, seq_len, batch_size)
end

using Flux, Functors, Test, NNlib, MLUtils

mha = MultiHeadAttention(64, 8)
sz = (64, 100, 32)
x = rand(Float32, sz)
y = mha(x)
@test y isa Array{Float32, 3}
@test size(y) == sz
