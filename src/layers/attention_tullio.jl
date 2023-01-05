using CUDAKernels, KernelAbstractions, LoopVectorization, Tullio

reshape_heads(x, nheads) = reshape(x, size(x, 1) ÷ nheads, nheads, size(x)[2:end]...)
flatten_heads(x) = reshape(x, :, size(x)[3:end]...)

function dot_product_attention_tullio(nheads::Int, q::A3, k::A3, v::A3; kws...)
  q, k, v = reshape_heads.((q, k, v), nheads)
  x, α = dot_product_attention_tullio(q, k, v; kws...)
  return flatten_heads(x), α
end


# Inspired by https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.dot_product_attention.html
function dot_product_attention_tullio(q::A4, k::A4, v::A4; 
            dropout=nothing, bias=nothing, mask=nothing)

  α = dot_product_attention_weights_tullio(q, k; dropout, bias, mask)
  # [α] = [kv_len, q_len, nheads, batch_size]
  @tullio x[d, h, i, b] := α[j, i, h, b] * v[d, h, j, b]
  # [x] = [kv_dim ÷ nheads, nheads, q_len, batch_size]
  return x, α
end

function dot_product_attention_weights_tullio(q::A4{T}, k::A4{T}; 
            dropout=nothing, mask=nothing, bias=nothing) where T

  q  = q ./ √T(size(q, 1))
  @tullio α[j, i, h, b] := q[d, h, i, b] * k[d, h, j, b]
  # [α] = [kv_len, q_len, nheads, batch_size]

  if bias !== nothing
    α = α .+ bias
  end
  if mask !== nothing
    neginf = typemin(eltype(α))
    α = ifelse.(mask, α, neginf)
  end

  α = softmax(α, dims=1)
  return dropout === nothing ? α : dropout(α)
end
