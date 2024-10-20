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

MLDataDevices.isleaf(::AMDGPU_CONV) = true
MLDataDevices.isleaf(::CPU_CONV) = true

_other_args(m::Conv) = (m.stride, m.pad, m.dilation, m.groups)
_other_args(m::ConvTranspose) = (m.stride, m.pad, m.outpad, m.dilation, m.groups)

# CPU -> GPU

function Adapt.adapt_structure(to::AMDGPUDevice, m::CPU_CONV)
    flipped_weight = reverse(m.weight; dims=ntuple(i -> i, ndims(m.weight) - 2))
    _conv_basetype(m)(
        Adapt.adapt(to, m.σ),
        Adapt.adapt(to, flipped_weight),
        Adapt.adapt(to, m.bias),
        _other_args(m)...)
end

# Don't adapt again.

Adapt.adapt_structure(to::AMDGPUDevice, m::AMDGPU_CONV) = m

# GPU -> CPU

function Adapt.adapt_structure(to::CPUDevice, m::AMDGPU_CONV)
    dims = ntuple(i -> i, ndims(m.weight) - 2)
    _conv_basetype(m)(
        Adapt.adapt(to, m.σ), reverse(Adapt.adapt(to, m.weight); dims),
        Adapt.adapt(to, m.bias), _other_args(m)...)
end
