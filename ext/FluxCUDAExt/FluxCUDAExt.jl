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

"""
    FluxCUDAExt.FluxCUDADevice <: Flux.AbstractDevice

A type representing `device` objects for the `"CUDA"` backend for Flux.
"""
Base.@kwdef struct FluxCUDADevice <: Flux.AbstractDevice
    deviceID::CUDA.CuDevice
end

(::FluxCUDADevice)(x) = gpu(FluxCUDAAdaptor(), x)
Flux._isavailable(::FluxCUDADevice) = true
Flux._isfunctional(::FluxCUDADevice) = CUDA.functional()
Flux._get_device_name(::FluxCUDADevice) = "CUDA"

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
    Flux.DEVICES[][Flux.GPU_BACKEND_ORDER["CUDA"]] = FluxCUDADevice(CUDA.device())

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
