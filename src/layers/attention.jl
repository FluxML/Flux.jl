using Flux, Functors, Test, LinearAlgebra, Random, Statistics
using CUDA
using CUDAKernels, KernelAbstractions, LoopVectorization, Tullio
using NeuralAttentionlib
using NeuralAttentionlib: score_returning
using BenchmarkTools
using Flux: glorot_uniform
using MLUtils
using ChainRulesCore
CUDA.allowscalar(false)

const A3{T} = AbstractArray{T, 3}
const A4{T} = AbstractArray{T, 4}

"""
    MultiHeadAttention(dims, num_heads; 
              [bias, init, attn_dropout_prob, proj_dropout_prob])

Multi-head dot-product attention layer.

# Arguments

- `dims`: ...
- `num_heads`: number of heads.
- `init`: weight initializer for the Dense layers.
- `bias` : whether pointwise QKVO dense transforms use bias.
- `attn_dropout_prob`: dropout probability after the self-attention layer
- `proj_dropout_prob`: dropout probability after the projection layer

# Forward
    
    (::MultiHeadAttention)(q_in, k_in, v_in; [mask, with_weights])

- `q_in`: input array of size `( seq_len, dims)
- `k_in`: input array of size `( seq_len, dims)
- `v_in`: input array of size `( seq_len, dims)
- `mask`: input array broadcastable to size 
   `(kv_len, q_len, num_heads, batch_size)`. Default `nothing`.
- `with_weights`: Whether to return the attention weights. Default `false`.

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
                     init = glorot_uniform,                    
                     attn_dropout_prob = 0.0, 
                     out_proj_dropout_prob = 0.0)

  dims = mha_process_dims(dims)
  @assert dims.qk % num_heads == 0 "qk_dim should be divisible by num_heads"
  qkv_proj = QKVProj(dims; bias, init)
  attn_drop = Dropout(attn_dropout_prob)
  out_proj = Chain(Dense(dims.v => dims.out; bias, init), Dropout(out_proj_dropout_prob))
  return MultiHeadAttention(num_heads, qkv_proj, attn_drop, out_proj)
end

# The following inputs are equivalent:
#  8 
#  8 => 8 => 8
#  (8, 8, 8) => 8 => 8
#  8 => (8, 8) => 8
#  (8, 8, 8) => (8, 8) => 8  # (q_in, k_in, v_in) => (qk, v) => out
mha_process_dims(dims::Int) = 
  (; q_in=dims, k_in=dims, v_in=dims, qk=dims, v=dims, out=dims)

const TuplInt2 = Union{Int, Tuple{Int, Int}}
const TuplInt3 = Union{Int, Tuple{Int, Int, Int}}

function mha_process_dims((in, (qkv, out))::Pair{<:TuplInt3, <:Pair{<:TuplInt2, Int}})
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
(m::MultiHeadAttention)(qkv; kws...) = m(qkv, qkv, qkv; kws...)

# key and value are the same
(m::MultiHeadAttention)(q, kv; kws...) = m(q, kv, kv; kws...)

function (m::MultiHeadAttention)(q_in::A3, k_in::A3, v_in::A3; 
      with_weights=false, mask=nothing, impl=:tullio)
  ## [q_in] = [q_in_dim, q_len, batch_size]
  ## [k_in] = [k_in_dim, kv_len, batch_size] 
  ## [v_in] = [v_in_dim, kv_len, batch_size]

  q, k, v = m.qkv_proj(q_in, k_in, v_in)
  # [q] = [qk_dim, q_len, batch_size]
  # [k] = [qk_dim, kv_len, batch_size]
  # [v] = [v_dim, kv_len, batch_size]

  if impl == :tullio
    x, α = dot_product_attention(m.num_heads, q, k, v; mask, dropout=m.attn_drop)
  elseif impl == :nalib
    x, α = NeuralAttentionlib.multihead_qkv_attention(score_returning, m.num_heads, q, k, v, mask)
  else
    error("Unknown attention implementation")
  end

  x = m.out_proj(x)

  return with_weights ? (x, α) : x
end

reshape_heads(x, num_heads) = reshape(x, size(x, 1) ÷ num_heads, num_heads, size(x)[2:end]...)
flatten_heads(x) = reshape(x, :, size(x)[3:end]...)

function dot_product_attention(num_heads::Int, q::A3, k::A3, v::A3; kws...)
  q, k, v = reshape_heads.((q, k, v), num_heads)
  x, α = dot_product_attention(q, k, v; kws...)
  return flatten_heads(x), α
end

# Inspired by https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.dot_product_attention.html
function dot_product_attention(q::A4, k::A4, v::A4; 
            dropout=nothing, bias=nothing, mask=nothing)

  α = dot_product_attention_weights(q, k; dropout, bias, mask)
  # [α] = [kv_len, q_len, num_heads, batch_size]
  @tullio x[d, h, i, b] := α[j, i, h, b] * v[d, h, j, b]
  # [x] = [kv_dim ÷ num_heads, num_heads, q_len, batch_size]
  return x, α
end

function dot_product_attention_weights(q::A4{T}, k::A4{T}; 
            dropout=nothing, mask=nothing, bias=nothing) where T

  q  = q ./ T(√size(q, 1))
  @tullio α[j, i, h, b] := q[d, h, i, b] * k[d, h, j, b]
  # [α] = [kv_len, q_len, num_heads, batch_size]

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


struct QKVProj
  q_proj::Dense
  k_proj::Dense
  v_proj::Dense
end

@functor QKVProj

function QKVProj(dims; bias = false, init=glorot_uniform)
  return QKVProj(
        Dense(dims.q_in => dims.qk; bias, init),
        Dense(dims.k_in => dims.qk; bias, init),
        Dense(dims.v_in => dims.v; bias, init)
      )
end

function (proj::QKVProj)(q_in, k_in, v_in)
  return (proj.q_proj(q_in), proj.k_proj(k_in), proj.v_proj(v_in))
end

function make_causal_mask(x::A3)
  d, len, batch_size = size(x)
  mask = triu(trues_like(x, (len, len)))
  return mask
end

trues_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), true)
falses_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), false)

@non_differentiable make_causal_mask(x)
@non_differentiable trues_like(::Any...)
@non_differentiable falses_like(::Any...)

function perf(dim, len, batch_size, num_heads)
  mha = MultiHeadAttention(dim, num_heads)  
  x = rand(Float32, (dim, len, batch_size))

  println("tullio")
  @btime $mha($x, impl=:tullio);
  @btime gradient(m -> sum(m($x, impl=:tullio)), $mha);

  println("nalib")
  @btime $mha($x, $x, $x, impl=:nalib);
  @btime gradient(m -> sum(m($x, impl=:nalib)), $mha);
  
  if CUDA.functional()
    mha_gpu = mha |> gpu
    x_gpu = x |> gpu

    println("tullio - gpu")
    @btime $mha_gpu($x_gpu, impl=:tullio);
    @btime gradient(m -> sum(m($x_gpu, impl=:tullio)), $mha_gpu);

    println("nalib - gpu")
    @btime CUDA.@sync $mha_gpu($x_gpu, impl=:nalib);
    @btime CUDA.@sync gradient(m -> sum(m($x_gpu, impl=:nalib)), $mha_gpu);
  end
  return nothing
end

function test(dim, num_heads, len, batch_size)
  mha = MultiHeadAttention(dim, num_heads)  
  x = rand(Float32, (dim, len, batch_size))
  y, α = mha(x, impl=:tullio, with_weights=true)
  @test y isa Array{Float32, 3}
  @test size(y) == (dim, len, batch_size)
  @test α isa Array{Float32, 4}
  @test size(α) == (len, len, num_heads, batch_size)

  y2, α2 = mha(x, impl=:nalib, with_weights=true)
  @test size(y) == size(y2)
  @test y2 ≈ y
  @test size(α) == size(α2)
  @test α2 ≈ α

  mask = make_causal_mask(x)
  y3, α3 = mha(x; impl=:tullio, with_weights=true, mask)
  y4, α4 = mha(x, impl=:nalib, with_weights=true, mask=NeuralAttentionlib.CausalMask())
  @test y ≈ y2
  @test α ≈ α2

  if CUDA.functional()
    mha_gpu = mha |> gpu
    x_gpu = x |> gpu

    y_gpu = mha_gpu(x_gpu, impl=:tullio)
    y_gpu2 = mha_gpu(x_gpu, impl=:nalib)
    @test Array(y_gpu) ≈ Array(y_gpu2)
    @test Array(y_gpu) ≈ y
  end
  return nothing
end

test(4, 2, 3, 1)

perf(128, 8, 128, 32)
# tullio
#   5.475 ms (80 allocations: 7.25 MiB)
#   13.073 ms (1172 allocations: 18.18 MiB)
# tullio - 6 threads
#   4.818 ms (192 allocations: 7.26 MiB)
#   10.927 ms (1398 allocations: 18.19 MiB)
# nalib
#   6.040 ms (91 allocations: 7.75 MiB)
#   14.542 ms (696 allocations: 16.17 MiB)
# nalib - 6 threads
#   7.832 ms (187 allocations: 7.76 MiB)
#   29.823 ms (988 allocations: 16.19 MiB)
# tullio - gpu
#   147.746 μs (523 allocations: 24.59 KiB)
#   957.111 μs (2413 allocations: 127.88 KiB)
# nalib - gpu
#   165.109 μs (411 allocations: 18.05 KiB)
#   659.685 μs (1527 allocations: 86.09 KiB)

dim = 2; len = 3; batch_size = 1; num_heads = 1
mha = MultiHeadAttention(dim, num_heads)  
x = rand(Float32, (dim, len, batch_size))
mask = make_causal_mask(x)
y, α = mha(x; impl=:tullio, with_weights=true, mask)
y2, α2 = mha(x; impl=:nalib, with_weights=true, mask=NeuralAttentionlib.CausalMask())
