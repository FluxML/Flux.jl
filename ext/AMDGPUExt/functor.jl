struct FluxAMDAdaptor end

# Convert Float64 to Float32, but preserve Float16.
adapt_storage(::FluxAMDAdaptor, x::T) where T <: AbstractArray =
    isbits(x) ? x : ROCArray(x)
adapt_storage(::FluxAMDAdaptor, x::AbstractArray{T, N}) where {T <: AbstractFloat, N} =
    isbits(x) ? x : ROCArray{Float32, N}(x)
adapt_storage(::FluxAMDAdaptor, x::AbstractArray{Float16, N}) where N =
    isbits(x) ? x : ROCArray{Float16, N}(x)

adapt_storage(::FluxAMDAdaptor, x::Zygote.FillArrays.AbstractFill) =
    ROCArray(collect(x))
adapt_storage(::FluxAMDAdaptor, x::Zygote.OneElement) = ROCArray(collect(x))
adapt_storage(::FluxAMDAdaptor, x::Random.TaskLocalRNG) =
    AMDGPU.rocRAND.default_rng()
adapt_storage(::FluxAMDAdaptor, x::AMDGPU.rocRAND.RNG) = x
adapt_storage(::FluxAMDAdaptor, x::AbstractRNG) = error("""
    Cannot map RNG of type $(typeof(x)) to AMDGPU.
    AMDGPU execution only supports Random.default_rng().""")

# TODO adaptor for Conv

adapt_storage(::FluxCPUAdaptor, x::AMDGPU.rocRAND.RNG) = Random.default_rng()

function ChainRulesCore.rrule(::Type{Array}, x::ROCArray)
    Array(x), dx -> (NoTangent(), ROCArray(unthunk(dx)))
end

function ChainRulesCore.rrule(
    ::typeof(Adapt.adapt_storage), to::FluxCPUAdaptor, x::AMDGPU.AnyROCArray,
)
    adapt_storage(to, x), dx -> (
        NoTangent(), NoTangent(),
        adapt_storage(FluxAMDAdaptor(), unthunk(dx)))
end

function _amd(x)
    check_use_amdgpu()
    USE_AMDGPU[] ?
        fmap(x -> Adapt.adapt(FluxAMDAdaptor(), x), x; exclude=_isleaf) :
        x
end
