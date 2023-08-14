# Convert Float64 to Float32, but preserve Float16.
function adapt_storage(to::FluxMetalAdaptor, x::AbstractArray)
    if typeof(to.ordinal) <: Nothing
        if (typeof(x) <: AbstractArray{Float16, N} where N)
            N = length(size(x))
            return isbits(x) ? x : MtlArray{Float16, N}(x)
        elseif (typeof(x) <: AbstractArray{T, N} where {T <: AbstractFloat, N})
            N = length(size(x))
            return isbits(x) ? x : MtlArray{Float32, N}(x)
        else
            return isbits(x) ? x : MtlArray(x)
        end
    end

    old_device = Metal.current_device()

    if !(x isa MtlArray)
        Metal.device!(Metal.devices()[to.ordinal])
        if (typeof(x) <: AbstractArray{Float16, N} where N)
            N = length(size(x))
            x_new = isbits(x) ? x : MtlArray{Float16, N}(x)
        elseif (typeof(x) <: AbstractArray{T, N} where {T <: AbstractFloat, N})
            N = length(size(x))
            x_new = isbits(x) ? x : MtlArray{Float32, N}(x)
        else
            x_new = isbits(x) ? x : MtlArray(x)
        end
        Metal.device!(old_device)
        return x_new
    elseif Metal.device(x).registryID == Metal.devices()[to.ordinal].registryID
        return x
    else
        Metal.device!(Metal.devices()[to.ordinal])
        x_new = copy(x)
        Metal.device!(old_device)
        return x_new
    end
end

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


function _metal(ordinal::Union{Nothing, Int}, x)
    check_use_metal()
    USE_METAL[] || return x
    fmap(x -> Adapt.adapt(FluxMetalAdaptor(ordinal), x), x; exclude=_isleaf)
end

function Flux.get_device(::Val{:Metal}, ordinal::Int)
    old_device = Metal.current_device()
    Metal.device!(Metal.devices()[ordinal])
    device = Flux.FluxMetalDevice(Metal.device())
    Metal.device!(old_device)
    return device
end
