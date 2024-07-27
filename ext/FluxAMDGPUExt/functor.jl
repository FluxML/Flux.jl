# Convert Float64 to Float32, but preserve Float16.
function adapt_storage(to::FluxAMDGPUAdaptor, x::AbstractArray)
    if to.id === nothing
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

    old_id = AMDGPU.device_id(AMDGPU.device()) - 1     # subtracting 1 because ids start from 0

    if !(x isa ROCArray)
        AMDGPU.device!(AMDGPU.devices()[to.id + 1])    # adding 1 because ids start from 0
        if (typeof(x) <: AbstractArray{Float16, N} where N)
            N = length(size(x))
            x_new = isbits(x) ? x : ROCArray{Float16, N}(x)
        elseif (typeof(x) <: AbstractArray{T, N} where {T <: AbstractFloat, N})
            N = length(size(x))
            x_new = isbits(x) ? x : ROCArray{Float32, N}(x)
        else
            x_new = isbits(x) ? x : ROCArray(x)
        end
        AMDGPU.device!(AMDGPU.devices()[old_id + 1])
        return x_new
    elseif AMDGPU.device_id(AMDGPU.device(x)) == to.id
        return x
    else
        AMDGPU.device!(AMDGPU.devices()[to.id + 1])
        x_new = copy(x)
        AMDGPU.device!(AMDGPU.devices()[old_id + 1])
        return x_new
    end
end

adapt_storage(::FluxAMDGPUAdaptor, x::Zygote.FillArrays.AbstractFill) =
    ROCArray(collect(x))
adapt_storage(::FluxAMDGPUAdaptor, x::Zygote.OneElement) = ROCArray(collect(x))
adapt_storage(::FluxAMDGPUAdaptor, x::Random.TaskLocalRNG) = AMDGPU.rocrand_rng()
adapt_storage(::FluxAMDGPUAdaptor, x::AMDGPU.rocRAND.RNG) = x
adapt_storage(::FluxAMDGPUAdaptor, x::AbstractRNG) = error("""
    Cannot map RNG of type $(typeof(x)) to AMDGPU.
    AMDGPU execution only supports Random.default_rng().""")

adapt_storage(::FluxCPUAdaptor, x::AMDGPU.rocRAND.RNG) = Random.default_rng()

function ChainRulesCore.rrule(
    ::typeof(Adapt.adapt_storage), to::FluxCPUAdaptor, x::AMDGPU.AnyROCArray,
)
    adapt_storage(to, x), dx -> (
        NoTangent(), NoTangent(),
        adapt_storage(FluxAMDGPUAdaptor(), unthunk(dx)))
end

# Since MIOpen supports only cross-correlation as convolution,
# for the actual convolution, we flip horizontally and vertically the weights.
# Same for CPU -> GPU & GPU -> CPU movements.
# Note, that gradients are also flipped.

const FLUX_CONV{M} = Union{
    Flux.Conv{<:Any, <:Any, <:Any, <:M, <:Any},
    Flux.ConvTranspose{<:Any, <:Any, <:Any, <:M, <:Any}}
const CPU_CONV = FLUX_CONV{Array}
const AMDGPU_CONV = FLUX_CONV{ROCArray}

_conv_basetype(::Conv) = Conv
_conv_basetype(::ConvTranspose) = ConvTranspose

Flux._isleaf(::AMDGPU_CONV) = true

_exclude(x) = Flux._isleaf(x)
_exclude(::CPU_CONV) = true

function _amd(id::Union{Nothing, Int}, x)
    check_use_amdgpu()
    USE_AMDGPU[] || return x
    fmap(x -> Adapt.adapt(FluxAMDGPUAdaptor(id), x), x; exclude=_exclude)
end

_other_args(m::Conv) = (m.stride, m.pad, m.dilation, m.groups)
_other_args(m::ConvTranspose) = (m.stride, m.pad, m.outpad, m.dilation, m.groups)

# CPU -> GPU

function Adapt.adapt_structure(to::FluxAMDGPUAdaptor, m::CPU_CONV)
    flipped_weight = reverse(m.weight; dims=ntuple(i -> i, ndims(m.weight) - 2))
    _conv_basetype(m)(
        Adapt.adapt(to, m.σ),
        Adapt.adapt(to, flipped_weight),
        Adapt.adapt(to, m.bias),
        _other_args(m)...)
end

# Don't adapt again.

Adapt.adapt_structure(to::FluxAMDGPUAdaptor, m::AMDGPU_CONV) = m

# GPU -> CPU

function Adapt.adapt_structure(to::FluxCPUAdaptor, m::AMDGPU_CONV)
    dims = ntuple(i -> i, ndims(m.weight) - 2)
    _conv_basetype(m)(
        Adapt.adapt(to, m.σ), reverse(Adapt.adapt(to, m.weight); dims),
        Adapt.adapt(to, m.bias), _other_args(m)...)
end

function Flux.get_device(::Val{:AMDGPU}, id::Int)     # id should start from 0
    old_id = AMDGPU.device_id(AMDGPU.device()) - 1     # subtracting 1 because ids start from 0
    AMDGPU.device!(AMDGPU.devices()[id + 1])           # adding 1 because ids start from 0
    device = Flux.FluxAMDGPUDevice(AMDGPU.device())
    AMDGPU.device!(AMDGPU.devices()[old_id + 1])
    return device
end
