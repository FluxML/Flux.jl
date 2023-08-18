# Convert Float64 to Float32, but preserve Float16.
adapt_storage(::FluxMetalAdaptor, x::T) where T <: AbstractArray =
    isbits(x) ? x : MtlArray(x)
adapt_storage(::FluxMetalAdaptor, x::AbstractArray{T, N}) where {T <: AbstractFloat, N} =
    isbits(x) ? x : MtlArray{Float32, N}(x)
adapt_storage(::FluxMetalAdaptor, x::AbstractArray{Float16, N}) where N =
    isbits(x) ? x : MtlArray{Float16, N}(x)

adapt_storage(::FluxMetalAdaptor, x::Zygote.FillArrays.AbstractFill) =
    MtlArray(collect(x))
adapt_storage(::FluxMetalAdaptor, x::Zygote.OneElement) = MtlArray(collect(x))
adapt_storage(::FluxMetalAdaptor, x::Random.TaskLocalRNG) =
    Metal.GPUArrays.default_rng(MtlArray)
adapt_storage(::FluxMetalAdaptor, x::Metal.GPUArrays.RNG) = x
adapt_storage(::FluxMetalAdaptor, x::AbstractRNG) = error("""
    Cannot map RNG of type $(typeof(x)) to Metal.
    Metal execution only supports Random.default_rng().""")

adapt_storage(::FluxCPUAdaptor, x::Metal.GPUArrays.RNG) = Random.default_rng()

function ChainRulesCore.rrule(
    ::typeof(Adapt.adapt_storage), to::FluxCPUAdaptor, x::MtlArray,
)
    adapt_storage(to, x), dx -> (
        NoTangent(), NoTangent(),
        adapt_storage(FluxMetalAdaptor(), unthunk(dx)))
end


function _metal(x)
    check_use_metal()
    USE_METAL[] || return x
    fmap(x -> Adapt.adapt(FluxMetalAdaptor(), x), x; exclude=_isleaf)
end
