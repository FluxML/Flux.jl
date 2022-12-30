using Flux, Functors, Test, LinearAlgebra, Random, Statistics
using CUDA, CUDAKernels, KernelAbstractions, LoopVectorization
using Tullio
using NeuralAttentionlib
using BenchmarkTools
CUDA.allowscalar(false)
const A3{T} = AbstractArray{T, 3}

"""
    MultiHeadAttention(dims, num_heads; 
              [bias, init, attn_dropout_prob, proj_dropout_prob])

Multi-head dot-product attention layer.

# Arguments

- `dims`: ...
- `nheads`: number of heads
- `init`: weight initializer for the Dense layers.
- `bias` : whether pointwise QKVO dense transforms use bias.
- `attn_dropout_prob`: dropout probability after the self-attention layer
- `proj_dropout_prob`: dropout probability after the projection layer

# Forward

- `in_q`: input tensor of shape `(batch_size, seq_len, dims)
- `in_k`: input tensor of shape `(batch_size, seq_len, dims)
- `in_v`: input tensor of shape `(batch_size, seq_len, dims)
- `mask`: input tensor of shape `(batch_size, seq_len, seq_len)`
- `return_weights`: whether to return the attention weights

# Examples

```julia
mha = MultiHeadAttention(64, 8)
```
"""
struct MultiHeadAttention
  num_heads::Int
  qkv_proj
  attn_drop
  out_proj
end

@functor MultiHeadAttention

function MultiHeadAttention(dims, num_heads::Int; 
                     bias::Bool = false,
                    #  init = glorot_uniform, # TODO
                     attn_dropout_prob = 0.0, 
                     out_proj_dropout_prob = 0.0)

  dims = mha_process_dims(dims)
  @assert dims.qkv % num_heads == 0 "qkv_dim should be divisible by num_heads"
  qkv_proj = QKVProj((dims.q_in, dims.k_in, dims.v_in) => dims.qkv; bias)
  attn_drop = Dropout(attn_dropout_prob)
  out_proj = Chain(Dense(dims.qkv => dims.out; bias), Dropout(out_proj_dropout_prob))
  return MultiHeadAttention(num_heads, qkv_proj, attn_drop, out_proj)
end

mha_process_dims(dims::Int) = (; q_in = dims, k_in = dims, v_in = dims, qkv = dims, out = dims)
mha_process_dims((in, (qkv, out))::Pair{Int, <:Pair}) = (; q_in = in, k_in = in, v_in = in, qkv, out)
mha_process_dims((in, (qkv, out))::Pair{<:Tuple, <:Pair}) = (; q_in = in[1], k_in = in[2], v_in = in[3], qkv, out)

# self-attention
(m::MultiHeadAttention)(x; kws...) = m(x, x, x; kws...)

function (m::MultiHeadAttention)(q_in::A3, k_in::A3, v_in::A3; with_weights=false, v=:tullio)
  ## [q_in] = [q_in_dim, q_len, batch_size]
  ## [k_in] = [k_in_dim, kv_len, batch_size] 
  ## [v_in] = [v_in_dim, kv_len, batch_size]

  if v == :tullio
    q, k, v = m.qkv_proj(q_in, k_in, v_in, m.num_heads)
    # [q] = [qkv_dim / num_heads, num_heads, q_len, batch_size]
    # [k] = [v] = [qkv_dim / num_heads, num_heads, kv_len, batch_size]
    
    x, α = dot_product_attention(q, k, v; dropout=m.attn_drop)
    x = reshape(x, :, size(x, 3), size(x, 4))
  elseif v == :nnalib
    q, k, v = m.qkv_proj(q_in, k_in, v_in)
    x = NeuralAttentionlib.multihead_qkv_attention(m.num_heads, q, k, v)
  else
    error("Unknown attention implementation")
  end

  x = m.out_proj(x)

  return x
  # return with_weights ? (x, α) : x
end

# Inspired by https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.dot_product_attention.html?highlight=dot_product_attention
function dot_product_attention(q, k, v; dropout=nothing)
  α = dot_product_attention_weights(q, k; dropout)
  # [α] = [kv_len, q_len, num_heads, batch_size]
  @tullio x[d, h, i, b] := α[j, i, h, b] * v[d, h, j, b]
  # [x] = [kv_dim ÷ num_heads, num_heads, q_len, batch_size]
  
  return x, α
end

function dot_product_attention_weights(q, k; dropout=nothing)
  @tullio α[j, i, h, b] := q[d, h, i, b] * k[d, h, j, b]
  # [α] = [kv_len, q_len, num_heads, batch_size]
  α = softmax(α, dims=1)
  return dropout === nothing ? α : dropout(α)
end


struct QKVProj
  k_proj::Dense
  v_proj::Dense
  q_proj::Dense
end

@functor QKVProj

function QKVProj((in_dim, qkv_dim)::Pair; bias = false)
  q_in_dim, k_in_dim, v_in_dim = in_dim
  return QKVProj(
      Dense(k_in_dim => qkv_dim; bias),
      Dense(v_in_dim => qkv_dim; bias),
      Dense(q_in_dim => qkv_dim; bias)
  )
end

function (proj::QKVProj)(q_in, k_in, v_in, num_heads)
  q = proj.q_proj(q_in)
  sz = size(q)
  newsz = (sz[1] ÷ num_heads, num_heads, sz[2:end]...)
  q = reshape(q, newsz)
  k = reshape(proj.k_proj(k_in), newsz)
  v = reshape(proj.v_proj(v_in), newsz)
  return q, k, v
end

function (proj::QKVProj)(q_in, k_in, v_in)
  return (proj.q_proj(q_in), proj.k_proj(k_in), proj.v_proj(v_in))
end


function perf(dim, len, batch_size, num_heads)
  mha = MultiHeadAttention(dim, num_heads)  
  x = rand(Float32, (dim, len, batch_size))

  println("tullio")
  @btime $mha($x, v=:tullio);
  @btime gradient(m -> sum(m($x, v=:tullio)), $mha);

  println("nnalib")
  @btime $mha($x, $x, $x, v=:nnalib);
  @btime gradient(m -> sum(m($x, v=:nnalib)), $mha);
  
  if CUDA.functional()
    mha_gpu = mha |> gpu
    x_gpu = x |> gpu

    println("tullio - gpu")
    @btime $mha_gpu($x_gpu, v=:tullio);
    @btime gradient(m -> sum(m($x_gpu, v=:tullio)), $mha_gpu);

    println("nnalib - gpu")
    @btime CUDA.@sync $mha_gpu($x_gpu, v=:nnalib);
    @btime CUDA.@sync gradient(m -> sum(m($x_gpu, v=:nnalib)), $mha_gpu);
  end
  return nothing
end

function test(dim, len, batch_size, num_heads)
  mha = MultiHeadAttention(dim, num_heads)  
  x = rand(Float32, (dim, len, batch_size))
  y = mha(x, v=:tullio)
  @test y isa Array{Float32, 3}
  @test size(y) == (dim, len, batch_size)
  y2 = mha(x, v=:nnalib)
  @test size(y) == size(y2)
  @test y2 ≈ y
  
  if CUDA.functional()
    mha_gpu = mha |> gpu
    x_gpu = x |> gpu

    y_gpu = mha_gpu(x_gpu, v=:tullio)
    y_gpu2 = mha_gpu(x_gpu, v=:nnalib)
    @test Array(y_gpu) ≈ Array(y_gpu2)
    @test Array(y_gpu) ≈ y
  end
  return nothing
end


test(12, 3, 2, 4)

perf(64, 100, 32, 4)
