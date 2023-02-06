
struct FluxCUDAAdaptor end

adapt_storage(to::FluxCUDAAdaptor, x) = CUDA.cu(x)
adapt_storage(to::FluxCUDAAdaptor, x::Zygote.FillArrays.AbstractFill) = CUDA.cu(collect(x))
if VERSION >= v"1.7"
  adapt_storage(to::FluxCUDAAdaptor, x::Random.TaskLocalRNG) = CUDA.default_rng()
else
  adapt_storage(to::FluxCUDAAdaptor, x::Random._GLOBAL_RNG) = CUDA.default_rng()
end
adapt_storage(to::FluxCUDAAdaptor, x::CUDA.RNG) = x
adapt_storage(to::FluxCUDAAdaptor, x::AbstractRNG) =
  error("Cannot map RNG of type $(typeof(x)) to GPU. GPU execution only supports Random.default_rng().")

# TODO: figure out the correct design for OneElement
adapt_storage(to::FluxCUDAAdaptor, x::Zygote.OneElement) = CUDA.cu(collect(x))

adapt_storage(to::FluxCPUAdaptor, x::T) where T <: CUDA.CUSPARSE.CUDA.CUSPARSE.AbstractCuSparseMatrix = adapt(Array, x)

# PIRACY
adapt_storage(to::FluxCPUAdaptor, x::CUDA.RNG) = Random.default_rng()

function ChainRulesCore.rrule(::typeof(Adapt.adapt_storage), to::FluxCPUAdaptor, x::CUDA.AbstractGPUArray)
  adapt_storage(to, x), dx -> (NoTangent(), NoTangent(), adapt_storage(FluxCUDAAdaptor(), unthunk(dx)))
end
  

# The following rrules for adapt are here to avoid double wrapping issues
# as seen in https://github.com/FluxML/Flux.jl/pull/2117#discussion_r1027321801

ChainRulesCore.rrule(::typeof(adapt), a::FluxCPUAdaptor, x::AnyCuArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), adapt(FluxCUDAAdaptor(), unthunk(Δ)))

ChainRulesCore.rrule(::typeof(adapt), a::FluxCUDAAdaptor, x::AnyCuArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), Δ)

ChainRulesCore.rrule(::typeof(adapt), a::FluxCUDAAdaptor, x::AbstractArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), adapt(FluxCPUAdaptor(), unthunk(Δ)))



"""
    gpu(x)

Moves `m` to the current GPU device, if available. It is a no-op otherwise.
See the [CUDA.jl docs](https://juliagpu.github.io/CUDA.jl/stable/usage/multigpu/) 
to help identify the current device.

This works for functions, and any struct marked with [`@functor`](@ref).

```julia-repl
julia> m = Dense(1,2)
Dense(1, 2)

julia> typeof(m.W)
Matrix{Float32}

julia> m_gpu = gpu(m)
Dense(1, 2)

julia> typeof(m_gpu.W) # notice the type of the array changed to a CuArray
CuArray{Float32, 2}
```
"""
function Flux.gpu(x)
  check_use_cuda()
  use_cuda[] ? fmap(x -> adapt(FluxCUDAAdaptor(), x), x; exclude = _isleaf) : x
end

function check_use_cuda()
  if use_cuda[] === nothing
    use_cuda[] = CUDA.functional()
    if use_cuda[] && !cuDNN.has_cudnn()
      @warn "CUDA.jl found cuda, but did not find libcudnn. Some functionality will not be available."
    end
    if !(use_cuda[])
      @info """The GPU function is being called but the GPU is not accessible. 
               Defaulting back to the CPU. (No action is required if you want to run on the CPU).""" maxlog=1
    end
  end
end

ChainRulesCore.@non_differentiable check_use_cuda()

