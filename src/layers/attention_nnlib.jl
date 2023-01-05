using ChainRulesCore
using NNlib

const AA3{T} = AbstractArray{T,3}
const AA4{T} = AbstractArray{T,4}
const AA{N,T} = AbstractArray{T,N}

"""
    dot_product_attention(query, key, value, [bias]; [fdrop, mask, nheads])

Multihead dot product attention used in transformer architectures. 

The input arrays must have the first two dimensions given by the number of features
and the sequece length, then an arbitrary number of batch dimensions or none.

Returns the attention output array of size `(v_dim, q_len, batch_size...)` and the attention scores.
of size `(kv_len, q_len, nheads, batch_size...)`.

See also [`dot_product_attention_scores`](@ref) if you only need the attention scores.

# Arguments

- `query`: Query array of size `(qk_dim, q_len, batch_size...)`.
- `key`: Key array of size `(qk_dim, kv_len, batch_size...)`.
- `value`: Value array of size `(v_dim, kv_len, batch_size...)`.
- `bias`: Either `nothing` or an array broadcastable to size `(kv_len, q_len, nheads, batch_size)`.
          It will be added to the attention scores before applying the softmax. Default `nothing`.
- `fdrop`: A dropout function or layer to be applied on the attention scores right after the softmax. 
           Default `identity` (no dropout). 
- `mask`: Either `nothing` or a boolean array broadcastable to size `(kv_len, q_len, nheads, batch_size)`.
          The mask is applied to the attention scores before the softmax.
          Can also be set to `mask=:causal` to apply a causal mask. Default `nothing`.
- `nheads`: Number of heads to split the input arrays into. Default `1`.

# Examples
    
```julia
q, k, v = rand(10, 20, 2), rand(10, 30, 2), rand(20, 30, 2)
y, α = dot_product_attention(q, k, v)
```
"""
function dot_product_attention(q::AA{N}, k::AA{N}, v::AA{N}, args...; kws...) where N
    batch_size = size(q)[3:end]
    batch_size == size(k)[3:end] == size(v)[3:end] || throw(ArgumentError("Batch dimensions have to be the same."))
    q, k, v = map(x -> reshape(x, size(x, 1), size(x, 2), :), (q, k, v))

    x, α = dot_product_attention(q, k, v, args...; kws...)

    x = reshape(x, size(x, 1), size(x, 2), batch_size...)
    α = reshape(α, size(α)[1:3]..., batch_size...)
    return x, α
end

function dot_product_attention(q::AA3, k::AA3, v::AA3, bias=nothing; 
            fdrop=identity, mask=nothing, nheads=1)

    (size(q, 3) == size(k, 3) == size(v, 3)) || throw(ArgumentError("Batch dimensions have to be the same."))
    size(q, 1) == size(k, 1) || throw(ArgumentError("First dimension in query and key has to be the same."))
    size(k, 2) == size(v, 2) || throw(ArgumentError("Second dimension in key and value has to be the same."))
    
    # Multihead attention. TODO create fastpath for singlehead attention.
    q, k, v = split_heads.((q, k, v), nheads)
    x, α = _dot_product_attention(q, k, v, bias, fdrop, mask)
    return join_heads(x), α
end

function _dot_product_attention(q::AA4, k::AA4, v::AA4, bias, fdrop, mask)
    # [q] = [qk_dim ÷ nheads, nheads, q_len, batch_size]
    # [k] = [qk_dim ÷ nheads, nheads, kv_len, batch_size]
    # [v] = [v_dim ÷ nheads, nheads, kv_len, batch_size]

    α = dot_product_attention_scores(q, k, bias; fdrop, mask)
    # [α] = [kv_len, q_len, nheads, batch_size]
    
    # The following permutedims and batched_mul are equivalent to
    # @tullio x[d, h, i, b] := α[j, i, h, b] * v[d, h, j, b]
    vt = permutedims(v, (1, 3, 2, 4))
    x = batched_mul(vt, α)
    x = permutedims(x, (1, 3, 2, 4))
    # [x] = [kv_dim ÷ nheads, nheads, q_len, batch_size]
    return x, α
end

"""
    dot_product_attention_scores(query, key, [bias]; [fdrop, mask])

Return the attention scores for the [`dot_product_attention`](@ref).
Input arrays must have dimensions 
`(num_features ÷ nheads, nheads, sequence_length, batch_size)`.

See [`dot_product_attention`](@ref) for more details.
"""
function dot_product_attention_scores(q::AA4{T}, k::AA4{T}, bias=nothing; 
            fdrop=identity, mask=nothing) where T

    # The following permutedims and batched_mul are equivalent to
    # @tullio logits[j, i, h, b] := q[d, h, i, b] * k[d, h, j, b] / √T(qk_dim)
    kt = permutedims(k, (3, 1, 2, 4))
    qt = permutedims(q, (1, 3, 2, 4)) ./ √T(size(q, 1))
    logits = batched_mul(kt, qt)
    # [logits] = [kv_len, q_len, nheads, batch_size]

    if bias !== nothing
        logits = logits .+ bias
    end

    if mask !== nothing
        if mask === :causal
            mask = make_causal_mask(logits)
        end
        neginf = typemin(eltype(logits))
        logits = ifelse.(mask, logits, neginf)
    end

    α = softmax(logits, dims=1)
    return fdrop(α)
end

""" 
    make_causal_mask(x, dims=2)

Return a boolean square matrix `m` of the same type as `x` and of side `size(x, dims)`.
Its elements are set such that `m[i, j] == i ≤ j`. 

Can be used to mask the attention scores in [`dot_product_attention`](@ref).
"""
function make_causal_mask(x::AbstractArray; dims::Int=2)
  len = size(x, dims)
  mask = triu(trues_like(x, (len, len)))
  return mask
end

trues_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), true)
falses_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), false)

split_heads(x, nheads) = reshape(x, size(x, 1) ÷ nheads, nheads, size(x)[2:end]...)
join_heads(x) = reshape(x, :, size(x)[3:end]...)

@non_differentiable make_causal_mask(x)
@non_differentiable trues_like(::Any...)
@non_differentiable falses_like(::Any...)

function NNlib.batched_mul(x::AbstractArray{T1,N}, y::AbstractArray{T2,N}) where {T1,T2,N}
    batch_size = size(x)[3:end]
    @assert batch_size == size(y)[3:end] "batch size has to be the same for the two arrays."
    x2 = reshape(x, size(x, 1), size(x, 2), :)
    y2 = reshape(y, size(y, 1), size(y, 2), :)
    z = batched_mul(x2, y2)
    return reshape(z, size(z, 1), size(z, 2), batch_size...)
  end

