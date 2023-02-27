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

# Since MIOpen supports only cross-correlation as convolution,
# for the actual convolution, we flip horizontally and vertically the weights.
# Same for CPU -> GPU & GPU -> CPU movements.
# Note, that gradients are also flipped.

# CPU -> GPU

_conv_basetype(c::Type{C}) where C <: Conv = Conv
_conv_basetype(c::Type{C}) where C <: ConvTranspose = ConvTranspose

function adapt_storage(to::FluxAMDAdaptor, m::C) where C <: Union{Conv, ConvTranspose}
    flipped_weight = reverse(m.weight; dims=ntuple(i -> i, ndims(m.weight) - 2))
    _conv_basetype(C)(
        Adapt.adapt(to, m.σ),
        Adapt.adapt(to, flipped_weight),
        Adapt.adapt(to, m.bias),
        m.stride, m.pad, m.dilation, m.groups)
end

# Don't adapt again.
function adapt_storage(
    to::FluxAMDAdaptor, m::Conv{N, M, F, A, V},
) where {N, M, F, A <: ROCArray, V}
    return m
end

function adapt_storage(
    to::FluxAMDAdaptor, m::ConvTranspose{N, M, F, A, V},
) where {N, M, F, A <: ROCArray, V}
    return m
end

_amd(m::Union{Conv, ConvTranspose}) = adapt_storage(FluxAMDAdaptor(), m)

# GPU -> CPU

function Flux.cpu(m::Conv{N, M, F, A, V}) where {N, M, F, A <: ROCArray, V}
    adapt_storage(FluxCPUAdaptor(), m)
end

function Flux.cpu(m::ConvTranspose{N, M, F, A, V}) where {N, M, F, A <: ROCArray, V}
    adapt_storage(FluxCPUAdaptor(), m)
end

function adapt_storage(
    to::FluxCPUAdaptor, m::Conv{N, M, F, A, V},
) where {N, M, F, A <: ROCArray, V}
    dims = ntuple(i -> i, ndims(m.weight) - 2)
    Conv(
        Adapt.adapt(to, m.σ), reverse(Adapt.adapt(to, m.weight); dims),
        Adapt.adapt(to, m.bias), m.stride, m.pad, m.dilation, m.groups)
end

function adapt_storage(
    to::FluxCPUAdaptor, m::ConvTranspose{N, M, F, A, V},
) where {N, M, F, A <: ROCArray, V}
    dims = ntuple(i -> i, ndims(m.weight) - 2)
    ConvTranspose(
        Adapt.adapt(to, m.σ), reverse(Adapt.adapt(to, m.weight); dims),
        Adapt.adapt(to, m.bias), m.stride, m.pad, m.dilation, m.groups)
end
