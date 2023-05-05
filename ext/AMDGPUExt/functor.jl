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

_exclude(x) = _isleaf(x)
_exclude(::CPU_CONV) = true

function _amd(x)
    check_use_amdgpu()
    USE_AMDGPU[] || return x
    fmap(x -> Adapt.adapt(FluxAMDAdaptor(), x), x; exclude=_exclude)
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
