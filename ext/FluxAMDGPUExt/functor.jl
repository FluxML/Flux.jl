# Convert Float64 to Float32, but preserve Float16.
function adapt_storage(to::FluxAMDAdaptor, x::AbstractArray)
    if typeof(to.ordinal) <: Nothing
        if (typeof(x) <: AbstractArray{Float16, N} where N)
            N = length(size(x))
            return isbits(x) ? x : ROCArray{Float16, N}(x)
        elseif (typeof(x) <: AbstractArray{T, N} where {T <: AbstractFloat, N})
            N = length(size(x))
            return isbits(x) ? x : ROCArray{Float32, N}(x)
        else
            return isbits(x) ? x : ROCArray(x)
        end
    end

    old_ordinal = AMDGPU.device_id(AMDGPU.device()) - 1     # subtracting 1 because ordinals start from 0

    if !(x isa ROCArray)
        AMDGPU.device!(AMDGPU.devices()[to.ordinal + 1])    # adding 1 because ordinals start from 0
        if (typeof(x) <: AbstractArray{Float16, N} where N)
            N = length(size(x))
            x_new = isbits(x) ? x : ROCArray{Float16, N}(x)
        elseif (typeof(x) <: AbstractArray{T, N} where {T <: AbstractFloat, N})
            N = length(size(x))
            x_new = isbits(x) ? x : ROCArray{Float32, N}(x)
        else
            x_new = isbits(x) ? x : ROCArray(x)
        end
        AMDGPU.device!(AMDGPU.devices()[old_ordinal + 1])
        return x_new
    elseif AMDGPU.device_id(AMDGPU.device(x)) == to.ordinal
        return x
    else
        AMDGPU.device!(AMDGPU.devices()[to.ordinal + 1])
        x_new = copy(x)
        AMDGPU.device!(AMDGPU.devices()[old_ordinal + 1])
        return x_new
    end
end

adapt_storage(::FluxAMDAdaptor, x::Zygote.FillArrays.AbstractFill) =
    ROCArray(collect(x))
adapt_storage(::FluxAMDAdaptor, x::Zygote.OneElement) = ROCArray(collect(x))
adapt_storage(::FluxAMDAdaptor, x::Random.TaskLocalRNG) =
    AMDGPU.rocRAND.default_rng()
adapt_storage(::FluxAMDAdaptor, x::AMDGPU.rocRAND.RNG) = x
adapt_storage(::FluxAMDAdaptor, x::AbstractRNG) = error("""
    Cannot map RNG of type $(typeof(x)) to AMDGPU.
    AMDGPU execution only supports Random.default_rng().""")

adapt_storage(::FluxCPUAdaptor, x::AMDGPU.rocRAND.RNG) = Random.default_rng()

function ChainRulesCore.rrule(
    ::typeof(Adapt.adapt_storage), to::FluxCPUAdaptor, x::AMDGPU.AnyROCArray,
)
    adapt_storage(to, x), dx -> (
        NoTangent(), NoTangent(),
        adapt_storage(FluxAMDAdaptor(), unthunk(dx)))
end

# Since MIOpen supports only cross-correlation as convolution,
# for the actual convolution, we flip horizontally and vertically the weights.
# Same for CPU -> GPU & GPU -> CPU movements.
# Note, that gradients are also flipped.

const FLUX_CONV{M} = Union{
    Flux.Conv{<:Any, <:Any, <:Any, <:M, <:Any},
    Flux.ConvTranspose{<:Any, <:Any, <:Any, <:M, <:Any}}
const CPU_CONV = FLUX_CONV{Array}
const AMD_CONV = FLUX_CONV{ROCArray}

_conv_basetype(::Conv) = Conv
_conv_basetype(::ConvTranspose) = ConvTranspose

Flux._isleaf(::AMD_CONV) = true

_exclude(x) = Flux._isleaf(x)
_exclude(::CPU_CONV) = true

function _amd(ordinal::Union{Nothing, Int}, x)
    check_use_amdgpu()
    USE_AMDGPU[] || return x
    fmap(x -> Adapt.adapt(FluxAMDAdaptor(ordinal), x), x; exclude=_exclude)
end

# CPU -> GPU

function Adapt.adapt_structure(to::FluxAMDAdaptor, m::CPU_CONV)
    flipped_weight = reverse(m.weight; dims=ntuple(i -> i, ndims(m.weight) - 2))
    _conv_basetype(m)(
        Adapt.adapt(to, m.σ),
        Adapt.adapt(to, flipped_weight),
        Adapt.adapt(to, m.bias),
        m.stride, m.pad, m.dilation, m.groups)
end

# Don't adapt again.

Adapt.adapt_structure(to::FluxAMDAdaptor, m::AMD_CONV) = m

# GPU -> CPU

function Adapt.adapt_structure(to::FluxCPUAdaptor, m::AMD_CONV)
    dims = ntuple(i -> i, ndims(m.weight) - 2)
    _conv_basetype(m)(
        Adapt.adapt(to, m.σ), reverse(Adapt.adapt(to, m.weight); dims),
        Adapt.adapt(to, m.bias), m.stride, m.pad, m.dilation, m.groups)
end

function Flux.get_device(::Val{:AMD}, ordinal::Int)     # ordinal should start from 0
    old_ordinal = AMDGPU.device_id(AMDGPU.device())
    AMDGPU.device!(AMDGPU.devices()[ordinal + 1])       # adding 1 because ordinals start from 0
    device = Flux.FluxAMDDevice(AMDGPU.device())
    AMDGPU.device!(AMDGPU.devices()[old_ordinal + 1])
    return device
end
