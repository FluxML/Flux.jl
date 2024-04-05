module FluxAMDGPUExt

import ChainRulesCore
import ChainRulesCore: NoTangent
import Flux
import Flux: FluxCPUAdaptor, FluxAMDGPUAdaptor, _amd, adapt_storage, fmap
import Flux: DenseConvDims, Conv, ConvTranspose, conv, conv_reshape_bias
import NNlib

using AMDGPU
using Adapt
using Random
using Zygote

const MIOPENFloat = AMDGPU.MIOpen.MIOPENFloat

# Set to boolean on the first call to check_use_amdgpu
const USE_AMDGPU = Ref{Union{Nothing, Bool}}(nothing)

function (device::Flux.FluxAMDGPUDevice)(x)
    if device.deviceID === nothing
        Flux.gpu(Flux.FluxAMDGPUAdaptor(), x)
    else
        return Flux.gpu(Flux.FluxAMDGPUAdaptor(AMDGPU.device_id(device.deviceID) - 1), x)  # subtracting 1, because device_id returns a positive integer
    end
end

Flux._get_device_name(::Flux.FluxAMDGPUDevice) = "AMDGPU"
Flux._isavailable(::Flux.FluxAMDGPUDevice) = true
Flux._isfunctional(::Flux.FluxAMDGPUDevice) = AMDGPU.functional()

function check_use_amdgpu()
    if !isnothing(USE_AMDGPU[])
        return
    end

    USE_AMDGPU[] = AMDGPU.functional()
    if USE_AMDGPU[]
        if !AMDGPU.functional(:MIOpen)
            @warn "MIOpen is not functional in AMDGPU.jl, some functionality will not be available."
        end
    else
        @info """
        The AMDGPU function is being called but AMDGPU.jl is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return
end

ChainRulesCore.@non_differentiable check_use_amdgpu()

include("functor.jl")
include("batchnorm.jl")
include("conv.jl")

function __init__()
    Flux.AMDGPU_LOADED[] = true
end

# TODO
# fail early if input to the model is not on the device (e.g. on the host)
# otherwise we get very cryptic errors & segfaults at the rocBLAS level

end
