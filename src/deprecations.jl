
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

function params!(p::Zygote.Params, x, seen = IdSet())
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

function params(m...)
  @warn """`Flux.params(m...)` is deprecated. Use `Flux.trainable(model)` for parameter collection,
  and the explicit `gradient(m -> loss(m, x, y), model)` for gradient computation.""" maxlog=1
  ps = Params()
  params!(ps, m)
  return ps
end

macro functor(args...)
  @warn """The use of `Flux.@functor` is deprecated.
      Most likely, you should write `Flux.@layer MyLayer` which will add various convenience methods for your type,
      such as pretty-printing and use with Adapt.jl.
      However, this is not required. Flux.jl v0.15 uses Functors.jl v0.5, which makes exploration of most nested `struct`s
      opt-out instead of opt-in... so Flux will automatically see inside any custom struct definitions.
      If you really want to apply the `@functor` macro to a custom struct, use `Functors.@functor` instead.
      """ maxlog=1

  return Functors.functorm(args...)
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

