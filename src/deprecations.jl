
# v0.13 deprecations

# Channel notation: Changed to match Conv, but very softly deprecated!
# Perhaps change to @deprecate for v0.15, but there is no plan to remove these.
Dense(in::Integer, out::Integer, σ = identity; kw...) =
  Dense(in => out, σ; kw...)
Bilinear(in1::Integer, in2::Integer, out::Integer, σ = identity; kw...) =
  Bilinear((in1, in2) => out, σ; kw...)
Embedding(in::Integer, out::Integer; kw...) = Embedding(in => out; kw...)

RNNCell(in::Integer, out::Integer, σ = tanh; kw...) = RNNCell(in => out, σ; kw...)
LSTMCell(in::Integer, out::Integer; kw...) = LSTMCell(in => out; kw...)

GRUCell(in::Integer, out::Integer; kw...) = GRUCell(in => out; kw...)
GRUv3Cell(in::Integer, out::Integer; kw...) = GRUv3Cell(in => out; kw...)


#### v0.14 deprecations ###########################

@deprecate default_rng_value() Random.default_rng()


# Issue 2476, after ConvTranspose got a new field in 2462. Minimal fix to allow loading?
function loadmodel!(dst::ConvTranspose, src::NamedTuple{(:σ, :weight, :bias, :stride, :pad, :dilation, :groups)}; kw...)
  new_src = (; src.σ, src.weight, src.bias, src.stride, src.pad, dst.outpad, src.dilation, src.groups)
  loadmodel!(dst, new_src; kw...)
end

# `Conv`/`CrossCor` gained a `pad_mode` field in #2717. Allow loading `Flux.state`
# checkpoints written before that, which lack the trailing `pad_mode` entry.
function loadmodel!(dst::Conv, src::NamedTuple{(:σ, :weight, :bias, :stride, :pad, :dilation, :groups)}; kw...)
  new_src = (; src.σ, src.weight, src.bias, src.stride, src.pad, src.dilation, src.groups, dst.pad_mode)
  loadmodel!(dst, new_src; kw...)
end

function loadmodel!(dst::CrossCor, src::NamedTuple{(:σ, :weight, :bias, :stride, :pad, :dilation)}; kw...)
  new_src = (; src.σ, src.weight, src.bias, src.stride, src.pad, src.dilation, dst.pad_mode)
  loadmodel!(dst, new_src; kw...)
end

function get_device(; verbose::Bool=false)
  Base.depwarn("get_device() is deprecated. Use `gpu_device()` instead.", :get_device)
  return MLDataDevices.gpu_device()
end

function get_device(backend::String, idx::Int = 0)
  Base.depwarn("get_device(backend::String, idx::Int) is deprecated. Use `gpu_device(idx+1)` instead.", :get_device)
  if backend == "AMD"
      @warn "\"AMD\" backend is deprecated. Please use \"AMDGPU\" instead." maxlog=1
      backend = "AMDGPU"
  end
  if backend == "CPU"
      return cpu_device()
  else
      return gpu_device(idx+1, force=true)
  end
end

function supported_devices()
  Base.depwarn("`supported_devices()` is deprecated. Use `supported_gpu_backends()` instead.", :supported_devices)
  return MLDataDevices.supported_gpu_backends()
end

# This was previosly documented.
# As of v0.14.23 we silently deprecate it.
# Later we will deprecate it loudly and then remove it.
const GPU_BACKEND = @load_preference("gpu_backend", "CUDA")


# help out with https://github.com/chengchingwen/Transformers.jl/issues/201
const FluxCPUAdaptor = CPUDevice
const FluxCUDAAdaptor = CUDADevice
const FluxAMDGPUAdaptor = AMDGPUDevice
const FluxMetalAdaptor = MetalDevice

function reset!(x)
  Base.depwarn("reset!(m) is deprecated. You can remove this call as it is no more needed.", :reset!)
  return x
end

function params!(p::Zygote.Params, x, seen = Base.IdSet())
  if x isa AbstractArray{<:Number} && Functors.isleaf(x)
    return push!(p, x)
  elseif x in seen
    nothing
  else
    push!(seen, x)
    for child in trainable(x)
      params!(p, child, seen)
    end
  end
end

"""
    params(model)

Returns a `Zygote.Params` object containing all parameter arrays from the model.
This is deprecated!
This function was the cornerstone of how Flux used Zygote's implicit mode gradients,
but since Flux 0.13 we use explicit mode `gradient(m -> loss(m, x, y), model)` instead.
To collect all the parameter arrays for other purposes, use `Flux.trainables(model)`.
"""
function params(m...)
  @warn """`Flux.params(m...)` is deprecated. Use `Flux.trainable(model)` for parameter collection,
  and the explicit `gradient(m -> loss(m, x, y), model)` for gradient computation.""" maxlog=1
  ps = Params()
  params!(ps, m)
  return ps
end

macro functor(args...)
  @warn """The use of `Flux.@functor` is deprecated.
      Most likely, you should write `Flux.@layer MyLayer`\
      which will add various convenience methods for your type,\
      such as pretty-printing and use with Adapt.jl.
      However, this is not required. Flux.jl v0.15 uses Functors.jl v0.5,\
      which makes exploration of most nested `struct`s opt-out instead of opt-in...\
      so Flux will automatically see inside any custom struct definitions.
      If you really want to apply the `@functor` macro to a custom struct, use `Functors.@functor` instead.
      """ maxlog=1
  # From https://discourse.julialang.org/t/calling-a-macro-from-within-a-macro-revisited/19680
  return esc(:($Functors.@functor($(args...))))
end

# Allows caching of the parameters when params is called within gradient() to fix #2040.
# @non_differentiable params(m...)  # https://github.com/FluxML/Flux.jl/pull/2054
# That speeds up implicit use, and silently breaks explicit use.
# From @macroexpand Zygote.@non_differentiable params(m...) and https://github.com/FluxML/Zygote.jl/pull/1248
Zygote._pullback(::Zygote.Context{true}, ::typeof(params), m...) = params(m), _ -> nothing

include("optimise/Optimise.jl") ## deprecated Module

function Optimiser(rules...)
  @warn "`Flux.Optimiser(...)` has been removed, please call `OptimiserChain(...)`, exported by Flux from Optimisers.jl" maxlog=1
  OptimiserChain(rules...)
end
function ClipValue(val)
  @warn "`Flux.ClipValue(...)` has been removed, please call `ClipGrad(...)`, exported by Flux from Optimisers.jl" maxlog=1
  ClipGrad(val)
end

# TODO this friendly error should go in Optimisers.jl.
# remove after https://github.com/FluxML/Optimisers.jl/pull/181
function Optimisers.update!(opt::Optimisers.AbstractRule, model, grad)
  error("""Invalid input to `update!`.
     `update!(state, model, grad)` needs `state = Flux.setup(opt, model)`.
    """)
end

# This exists to solve an ambiguity between the method above & one in layers/basic.jl
function Optimisers.update!(opt::Optimisers.AbstractRule, model::Chain, grad::Tuple)
  error("""Invalid input to `update!`.
     `update!(state, model, grad)` needs `state = Flux.setup(opt, model)`.
    """)
end

# From 0.15, Flux.gradient is not Zygote.gradient, but we can add a deprecation path:
function gradient(f, p::Zygote.Params)
  Base.depwarn("""Implicit gradients such as `gradient(f, ::Params)` are deprecated in Flux!
    Please see the docs for new explicit form.""", :gradient; force=true)
  Zygote.gradient(f, p)
end
function withgradient(f, p::Zygote.Params)
  Base.depwarn("""Implicit gradients such as `withgradient(f, ::Params)` are deprecated in Flux!
    Please see the docs for new explicit form.""", :withgradient; force=true)
  Zygote.withgradient(f, p)
end

          
### v0.16 deprecations ####################



# train!(loss::Function, ps::Zygote.Params, data, opt) = throw(ArgumentError(
#   """On Flux 0.16, `train!` no longer accepts implicit `Zygote.Params`.
#   Instead of `train!(loss_xy, Flux.params(model), data, Adam())`
#   it now needs `opt = Flux.setup(Adam(), model); train!(loss_mxy, model, data, opt)`
#   where `loss_mxy` accepts the model as its first argument.
#   """
# ))

# In Flux <= v0.14, pooling layers contained no state.
# These workarounds allow such pooling layers to be loaded in Flux >= v0.16. 
# (https://github.com/FluxML/Flux.jl/pull/2598)
loadmodel!(dst::MaxPool{N, M}, src::Tuple{}; kw...) where {N, M} = dst
loadmodel!(dst::MeanPool{N, M}, src::Tuple{}; kw...) where {N, M} = dst
loadmodel!(dst::AdaptiveMaxPool{S, O}, src::Tuple{}; kw...) where {S, O} = dst
loadmodel!(dst::AdaptiveMeanPool{S, O}, src::Tuple{}; kw...) where {S, O} = dst


### Unreleased deprecations ####################

"""
    SamePad()

Deprecated alias for `pad=:same`, kept for backwards compatibility.
Prefer passing the symbol `:same` to the `pad` keyword of convolutional
and pooling layers. See [`Conv`](@ref).
"""
struct SamePad end

calc_padding(lt, ::SamePad, k::NTuple{N,T}, dilation, stride) where {N,T} =
  calc_padding(lt, Val(:same), k, dilation, stride)


# This method let you use Optimisers.Descent() without setup, when there is no state
function Train.train!(loss, model, data, rule::Optimisers.AbstractRule; cb = nothing, caching_allocator::Bool = false, gc_interval::Union{Integer, Symbol} = :auto)
    return train!(loss, model, data, _rule_to_state(model, rule); cb, caching_allocator, gc_interval)
end

# This method let you use Optimisers.Descent() without setup, when there is no state
function Train.train!(loss, model::EnzymeCore.Duplicated, data, rule::Optimisers.AbstractRule; cb=nothing, caching_allocator::Bool = false, gc_interval::Union{Integer, Symbol} = :auto)
    return train!(loss, model, data, _rule_to_state(model, rule); cb, caching_allocator, gc_interval)
end

function _rule_to_state(model, rule::Optimisers.AbstractRule)
    state = setup(rule, model)
    @gensym warn_id
    name = typeof(rule).name.name
    fmap(state, exclude = x -> x isa Optimisers.Leaf) do leaf
        leaf.state isa Nothing ||  @warn """Optimiser $name has state which will be discarded after `train!` finishes.
                                            Please run `opt = Flux.setup($name(), model)` and pass this `opt` to `train!`.""" leaf maxlog=1 _id=warn_id
        leaf
    end
    return state
end


"""
    MultiHeadAttention(dims; [nheads, bias, init, dropout_prob])

Deprecated: use [`Attention`](@ref) instead, which takes the same `dims` and returns only
the transformed input sequence, and [`Flux.attention_weights`](@ref) for the attention scores.
The keyword `dropout_prob` is called `dropout` there, and the attention `bias` is a keyword
argument of the forward pass. `Attention(mha)` converts a `MultiHeadAttention` layer.

The multi-head dot-product attention layer used in Transformer architectures.
Returns the transformed input sequence and the attention scores.

# Forward

    (mha::MultiHeadAttention)(q_in, k_in, v_in, [bias]; [mask])

The inputs have size `(features, len, batch_size)`, while `bias` and `mask` are broadcastable
to the size `(kv_len, q_len, nheads, batch_size)` of the returned attention scores.
Alternative calling signatures are `mha(q_in)`, equivalent to `mha(q_in, q_in, q_in)`,
and `mha(q_in, k_in)`, equivalent to `mha(q_in, k_in, k_in)`.
"""
struct MultiHeadAttention{P1, D, P2}
  nheads::Int
  q_proj::P1
  k_proj::P1
  v_proj::P1
  attn_drop::D
  out_proj::P2
end

@layer :noexpand MultiHeadAttention

function MultiHeadAttention(dims; 
                     nheads::Int = 8,
                     bias::Bool = false,
                     init = glorot_uniform,                    
                     dropout_prob = 0.0)

  Base.depwarn("`MultiHeadAttention` is deprecated, use `Attention` instead, which returns only \
    the output and not the attention scores (see `Flux.attention_weights` for those).", :MultiHeadAttention)
  dims = normalize_mha_dims(dims)
  dims.qk % nheads == 0 || throw(ArgumentError("qk_dim = $(dims.qk) should be divisible by nheads = $(nheads)"))
  dims.v % nheads == 0 || throw(ArgumentError( "v_dim = $(dims.v) should be divisible by nheads = $(nheads)"))
  q_proj = Dense(dims.q_in => dims.qk; bias, init)
  k_proj = Dense(dims.k_in => dims.qk; bias, init)
  v_proj = Dense(dims.v_in => dims.v; bias, init)
  attn_drop = Dropout(dropout_prob)
  out_proj = Dense(dims.v => dims.out; bias, init)
  return MultiHeadAttention(nheads, q_proj, k_proj, v_proj, attn_drop, out_proj)
end

# self-attention
(mha::MultiHeadAttention)(qkv; kws...) = mha(qkv, qkv, qkv; kws...)

# key and value are the same
(mha::MultiHeadAttention)(q, kv; kws...) = mha(q, kv, kv; kws...)

function (mha::MultiHeadAttention)(q_in::A3, k_in::A3, v_in::A3, 
                                  bias=nothing; mask=nothing)
  q = _split_heads(mha.q_proj(q_in), mha.nheads)  # [q] = [qk_head_dim, nheads, q_len, batch_size]
  k = _split_heads(mha.k_proj(k_in), mha.nheads)  # [k] = [qk_head_dim, nheads, kv_len, batch_size]
  v = _split_heads(mha.v_proj(v_in), mha.nheads)  # [v] = [v_head_dim, nheads, kv_len, batch_size]
  x, α = NNlib.scaled_dot_product_attention_with_weights(q, k, v, bias; mask, fdrop=mha.attn_drop)
  # [α] = [kv_len, q_len, nheads, batch_size]
  x = mha.out_proj(reshape(x, :, size(x)[3:end]...))
  # [x] = [out_dim, q_len, batch_size]
  return x, α
end

function Base.show(io::IO, mha::MultiHeadAttention)
  qk, q_in = size(mha.q_proj.weight)
  qk, k_in = size(mha.k_proj.weight)
  v, v_in = size(mha.v_proj.weight)
  out, v = size(mha.out_proj.weight)
  print(io, "MultiHeadAttention(")
  if q_in == k_in == v_in == qk == v == out
    print(io, q_in)
  elseif q_in == k_in == v_in && qk == v
    print(io, q_in, " => ", qk, " => ", out)
  elseif q_in == k_in == v_in
    print(io, q_in, " => (", qk, ", ", v,") => ", out)
  else
    print(io, "(", q_in, ", ", k_in, ", ", v_in, ") => (", qk, ", ", v,") => ", out)
  end
  print(io, "; nheads=", mha.nheads)
  if mha.q_proj.bias !== false
    print(io, ", bias=true")
  end
  if mha.attn_drop.p != 0
    print(io, ", dropout_prob=", mha.attn_drop.p)
  end
  print(io, ")")
end

# The converted layer shares its parameters with `mha`.
Attention(mha::MultiHeadAttention) =
  Attention(mha.nheads, mha.nheads, mha.q_proj, mha.k_proj, mha.v_proj, identity, identity,
            mha.attn_drop, mha.out_proj, false)

# Allow loading a `MultiHeadAttention` layer or its `Flux.state` into an `Attention` layer.
function loadmodel!(dst::Attention, src::NamedTuple{(:nheads, :q_proj, :k_proj, :v_proj, :attn_drop, :out_proj)}; kw...)
  new_src = (; src.nheads, dst.nkvheads, src.q_proj, src.k_proj, src.v_proj, dst.q_norm, dst.k_norm,
             src.attn_drop, src.out_proj, dst.is_causal)
  loadmodel!(dst, new_src; kw...)
end

loadmodel!(dst::Attention, src::MultiHeadAttention; kw...) =
  loadmodel!(dst, Functors.children(src); kw...)
