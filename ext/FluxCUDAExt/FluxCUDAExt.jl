module FluxCUDAExt

using Flux
import Flux: _cuda
using Flux: FluxCPUAdaptor, FluxCUDAAdaptor, fmap
using CUDA
using NNlib
using Zygote
using ChainRulesCore
using Random
using Adapt
import Adapt: adapt_storage


const USE_CUDA = Ref{Union{Nothing, Bool}}(nothing)

function (device::Flux.FluxCUDADevice)(x)
    if device.deviceID === nothing
        return Flux.gpu(Flux.FluxCUDAAdaptor(), x)
    else
        return Flux.gpu(Flux.FluxCUDAAdaptor(device.deviceID.handle), x)
    end
end
Flux._get_device_name(::Flux.FluxCUDADevice) = "CUDA"
Flux._isavailable(::Flux.FluxCUDADevice) = true
Flux._isfunctional(::Flux.FluxCUDADevice) = CUDA.functional()

function check_use_cuda()
    if !isnothing(USE_CUDA[])
        return
    end

    USE_CUDA[] = CUDA.functional()
    if !USE_CUDA[]
        @info """
        The CUDA function is being called but CUDA.jl is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return
end

ChainRulesCore.@non_differentiable check_use_cuda()

include("functor.jl")

function __init__()
    Flux.CUDA_LOADED[] = true

    ## add device to available devices
    Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["CUDA"]] = CUDA.functional() ? Flux.FluxCUDADevice(CUDA.device()) : Flux.FluxCUDADevice(nothing)

    try
       Base.require(Main, :cuDNN)
    catch
        @warn """Package cuDNN not found in current path.
        - Run `import Pkg; Pkg.add(\"cuDNN\")` to install the cuDNN package, then restart julia.
        - If cuDNN is not installed, some Flux functionalities will not be available when running on the GPU.
        """
    end
end

end 
