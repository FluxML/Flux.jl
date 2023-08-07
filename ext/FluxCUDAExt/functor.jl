
adapt_storage(to::FluxCUDAAdaptor, x) = CUDA.cu(x)
adapt_storage(to::FluxCUDAAdaptor, x::Zygote.FillArrays.AbstractFill) = CUDA.cu(collect(x))
adapt_storage(to::FluxCUDAAdaptor, x::Random.TaskLocalRNG) = CUDA.default_rng()
adapt_storage(to::FluxCUDAAdaptor, x::CUDA.RNG) = x
adapt_storage(to::FluxCUDAAdaptor, x::AbstractRNG) =
  error("Cannot map RNG of type $(typeof(x)) to GPU. GPU execution only supports Random.default_rng().")

# TODO: figure out the correct design for OneElement
adapt_storage(to::FluxCUDAAdaptor, x::Zygote.OneElement) = CUDA.cu(collect(x))

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

function _cuda(x)
  check_use_cuda()
  USE_CUDA[] || return x
  fmap(x -> Adapt.adapt(FluxCUDAAdaptor(), x), x; exclude=Flux._isleaf)
end

function Flux.get_device(::Type{CUDA.CUDABackend}, ordinal::UInt)
    CUDA.device!(ordinal) do
        return Flux.FluxCUDADevice(CUDA.device())
    end
end
