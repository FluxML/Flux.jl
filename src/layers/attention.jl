using Flux, Functors, Test, LinearAlgebra, Random, Statistics
using CUDA
using NeuralAttentionlib
using NeuralAttentionlib: score_returning
using BenchmarkTools
using Flux: glorot_uniform
CUDA.allowscalar(false)

const A3{T} = AbstractArray{T, 3}
const A4{T} = AbstractArray{T, 4}
const TuplInt2 = Union{Int, Tuple{Int, Int}}
const TuplInt3 = Union{Int, Tuple{Int, Int, Int}}

include("attention_nnlib.jl")
include("attention_tullio.jl")


"""
    MultiHeadAttention(dims, nheads; [bias, init, dropout_prob])

Multi-head dot-product attention layer.

# Arguments

- `dims`: ...
- `nheads`: number of heads.
- `init`: weight initializer for the Dense layers.
- `bias` : whether pointwise QKVO dense transforms use bias.
- `dropout_prob`: dropout probability for the attention scores.

# Forward
    
    (::MultiHeadAttention)(q_in, k_in, v_in, [bias]; [mask, withscores])

- `q_in`: input query array of size `(q_in_dim, q_len, batch_size...)`.
- `k_in`: input key array of size `(k_in_dim, kv_len, batch_size...)`.
- `v_in`: input value array of size `(v_in_dim, kv_len, batch_size...)`.
- `mask`: input array broadcastable to size 
          `(kv_len, q_len, nheads, batch_size)`. Default `nothing`.
- `withscores`: Whether to return the attention scores. Default `false`.

# Examples

```julia
mha = MultiHeadAttention(64, 8)
```
"""
struct MultiHeadAttention{P1, D, P2}
  nheads::Int
  qkv_proj::P1
  attn_drop::D
  out_proj::P2
end

@functor MultiHeadAttention

function MultiHeadAttention(dims, nheads::Int; 
                     bias::Bool = false,
                     init = glorot_uniform,                    
                     dropout_prob = 0.0)

  dims = mha_process_dims(dims)
  @assert dims.qk % nheads == 0 "qk_dim should be divisible by nheads"
  qkv_proj = QKVProj(dims; bias, init)
  attn_drop = Dropout(dropout_prob)
  out_proj = Dense(dims.v => dims.out; bias, init)
  return MultiHeadAttention(nheads, qkv_proj, attn_drop, out_proj)
end

mha_process_dims(dims::Int) = 
  (; q_in=dims, k_in=dims, v_in=dims, qk=dims, v=dims, out=dims)

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

function (m::MultiHeadAttention)(q_in::A3, k_in::A3, v_in::A3, bias=nothing; 
      withscores=false, mask=nothing, impl=:nnlib)
  ## [q_in] = [q_in_dim, q_len, batch_size]
  ## [k_in] = [k_in_dim, kv_len, batch_size] 
  ## [v_in] = [v_in_dim, kv_len, batch_size]

  q, k, v = m.qkv_proj(q_in, k_in, v_in)
  # [q] = [qk_dim, q_len, batch_size]
  # [k] = [qk_dim, kv_len, batch_size]
  # [v] = [v_dim, kv_len, batch_size]

  if impl == :tullio
    x, α = dot_product_attention_tullio(m.nheads, q, k, v; mask, dropout=m.attn_drop)
  elseif impl == :nalib
    x, α = NeuralAttentionlib.multihead_qkv_attention(score_returning, m.nheads, q, k, v, mask)
  elseif impl == :nnlib
    x, α = dot_product_attention(q, k, v, bias; m.nheads, mask, fdrop=m.attn_drop)
  else
    error("Unknown attention implementation")
  end

  x = m.out_proj(x)

  return withscores ? (x, α) : x
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
            Dense(dims.v_in => dims.v; bias, init))
end

function (proj::QKVProj)(q_in, k_in, v_in)
  return (proj.q_proj(q_in), proj.k_proj(k_in), proj.v_proj(v_in))
end

function perf(dim, len, batch_size, nheads)
  mha = MultiHeadAttention(dim, nheads)  
  x = rand(Float32, (dim, len, batch_size))

  println("tullio")
  @btime $mha($x, impl=:tullio);
  @btime gradient(m -> sum(m($x, impl=:tullio)), $mha);

  println("nalib")
  @btime $mha($x, $x, $x, impl=:nalib);
  @btime gradient(m -> sum(m($x, impl=:nalib)), $mha);

  println("nnlib")
  @btime $mha($x, $x, $x, impl=:nnlib);
  @btime gradient(m -> sum(m($x, impl=:nnlib)), $mha);
  
  if CUDA.functional()
    mha_gpu = mha |> gpu
    x_gpu = x |> gpu

    println("tullio - gpu")
    @btime $mha_gpu($x_gpu, impl=:tullio);
    @btime gradient(m -> sum(m($x_gpu, impl=:tullio)), $mha_gpu);

    println("nalib - gpu")
    @btime CUDA.@sync $mha_gpu($x_gpu, impl=:nalib);
    @btime CUDA.@sync gradient(m -> sum(m($x_gpu, impl=:nalib)), $mha_gpu);

    println("nnlib - gpu")
    @btime CUDA.@sync $mha_gpu($x_gpu, impl=:nnlib);
    @btime CUDA.@sync gradient(m -> sum(m($x_gpu, impl=:nnlib)), $mha_gpu);
  end
  return nothing
end

function test(dim, nheads, len, batch_size)
  mha = MultiHeadAttention(dim, nheads)
  q = rand(Float32, (dim, len, batch_size))
  k = rand(Float32, (dim, len, batch_size))
  v = rand(Float32, (dim, len, batch_size))
  
  y, α = mha(q, k, v, impl=:tullio, withscores=true)
  @test y isa Array{Float32, 3}
  @test size(y) == (dim, len, batch_size)
  @test α isa Array{Float32, 4}
  @test size(α) == (len, len, nheads, batch_size)

  y2, α2 = mha(q, k, v, impl=:nalib, withscores=true)
  @test size(y) == size(y2)
  @test y2 ≈ y
  @test size(α) == size(α2)
  @test α2 ≈ α

  y2b, α2b = mha(q, k, v, impl=:nnlib, withscores=true)
  @test size(y) == size(y2b)
  @test y2b ≈ y
  @test size(α) == size(α2b)
  @test α2b ≈ α

  mask = make_causal_mask(q)
  y3, α3 = mha(q, k, v; impl=:tullio, withscores=true, mask)
  y4, α4 = mha(q, k, v, impl=:nalib, withscores=true, mask=NeuralAttentionlib.CausalMask())
  @test y3 ≈ y4
  @test α3 ≈ α4

  if CUDA.functional()
    mha_gpu = mha |> gpu
    q_gpu, k_gpu, v_gpu = q |> gpu, k |> gpu, v |> gpu
    
    y_gpu = mha_gpu(q_gpu, k_gpu, v_gpu, impl=:tullio)
    y_gpu2 = mha_gpu(q_gpu, k_gpu, v_gpu, impl=:nalib)
    @test Array(y_gpu) ≈ Array(y_gpu2)
    @test Array(y_gpu) ≈ y
  end
  return nothing
end

test(4, 2, 3, 1)

perf(128, 8, 128, 32)

## M1 Pro, NNlib v0.8.12
# tullio
#   2.948 ms (77 allocations: 7.25 MiB)
#   15.041 ms (1124 allocations: 16.71 MiB)
# nalib
#   3.503 ms (89 allocations: 7.75 MiB)
#   15.828 ms (604 allocations: 14.70 MiB)
# nnlib
#   3.611 ms (87 allocations: 9.25 MiB)
#   16.497 ms (1055 allocations: 20.71 MiB)

## M1 Pro, NNlib v0.8.13 (fast_maximum)
# tullio
#   2.427 ms (71 allocations: 7.13 MiB)
#   14.510 ms (1118 allocations: 16.59 MiB)
# nalib
#   3.052 ms (84 allocations: 7.63 MiB)
#   15.327 ms (599 allocations: 14.57 MiB)
# nnlib
#   3.166 ms (81 allocations: 9.13 MiB)
#   16.082 ms (1049 allocations: 20.58 MiB)


# function prof()
  # dim, len, batch_size, nheads = 128, 8, 128, 32;
  # # dim = 384; len = 128; batch_size = 32; nheads = 12
  # mha = MultiHeadAttention(dim, nheads)  
  # x = rand(Float32, (dim, len, batch_size))
  # @btime mha(x, impl=:tullio);
  # @btime mha(x, impl=:nnlib);
  # @profview mha(x, impl=:tullio);
  # @profview prof(mha, x);
  # y, α = mha(x; impl=:nnlib, withscores=true, mask)
  # y2, α2 = mha(x; impl=:nalib, withscores=true, mask=NeuralAttentionlib.CausalMask())
# end