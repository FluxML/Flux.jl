adapt_storage(to::FluxCUDAAdaptor, x) = CUDA.cu(x)

function adapt_storage(to::FluxCUDAAdaptor, x::AbstractArray)
    to.id === nothing && return CUDA.cu(x)

    # remember current device
    old_id = CUDA.device().handle

    if !(x isa CuArray)
        CUDA.device!(to.id)
        x_new = CUDA.cu(x)
        CUDA.device!(old_id)
        return x_new
    elseif CUDA.device(x).handle == to.id
        return x
    else
        CUDA.device!(to.id)
        x_new = copy(x)
        CUDA.device!(old_id)
        return x_new
    end
end

adapt_storage(to::FluxCUDAAdaptor, x::Zygote.FillArrays.AbstractFill) = CUDA.cu(collect(x))
adapt_storage(to::FluxCUDAAdaptor, x::Random.TaskLocalRNG) = CUDA.default_rng()
adapt_storage(to::FluxCUDAAdaptor, x::CUDA.RNG) = x
adapt_storage(to::FluxCUDAAdaptor, x::AbstractRNG) =
  error("Cannot map RNG of type $(typeof(x)) to GPU. GPU execution only supports Random.default_rng().")

# TODO: figure out the correct design for OneElement
adapt_storage(to::FluxCUDAAdaptor, x::Zygote.OneElement) = CUDA.cu(collect(x))
# Patch for GPU support until we can make OneElement smarter
if isdefined(Zygote.ChainRules, :OneElement)
    adapt_storage(to::FluxCUDAAdaptor, x::Zygote.ChainRules.OneElement) = CUDA.cu(collect(x))
end

adapt_storage(to::FluxCPUAdaptor, x::T) where T <: CUDA.CUSPARSE.CUDA.CUSPARSE.AbstractCuSparseMatrix = adapt(Array, x)
adapt_storage(to::FluxCPUAdaptor, x::CUDA.RNG) = Random.default_rng()

function ChainRulesCore.rrule(::typeof(Adapt.adapt_storage), to::FluxCPUAdaptor, x::CUDA.AbstractGPUArray)
  adapt_storage(to, x), dx -> (NoTangent(), NoTangent(), adapt_storage(FluxCUDAAdaptor(), unthunk(dx)))
end

ChainRulesCore.rrule(::typeof(adapt), a::FluxCPUAdaptor, x::AnyCuArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), adapt(FluxCUDAAdaptor(), unthunk(Δ)))

ChainRulesCore.rrule(::typeof(adapt), a::FluxCUDAAdaptor, x::AnyCuArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), Δ)

ChainRulesCore.rrule(::typeof(adapt), a::FluxCUDAAdaptor, x::AbstractArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), adapt(FluxCPUAdaptor(), unthunk(Δ)))

function _cuda(id::Union{Nothing, Int}, x)
  check_use_cuda()
  USE_CUDA[] || return x
  fmap(x -> Adapt.adapt(FluxCUDAAdaptor(id), x), x; exclude=Flux._isleaf)
end

function Flux.get_device(::Val{:CUDA}, id::Int = -1)
  if id < 0 # return current device
    # return CUDA.functional() ? Flux.FluxCUDADevice(CUDA.device()) : Flux.FluxCUDADevice(nothing)
    return Flux.FluxCUDADevice(CUDA.device()) 
  else
    old_id = CUDA.device().handle
    CUDA.device!(id)
    device = Flux.FluxCUDADevice(CUDA.device())
    CUDA.device!(old_id)
    return device
  end
end
