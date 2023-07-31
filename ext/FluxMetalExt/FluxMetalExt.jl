module FluxMetalExt

import Flux
import Flux: FluxCPUAdaptor, FluxMetalAdaptor, _metal, _isleaf, adapt_storage, fmap
import NNlib
using ChainRulesCore

using Metal
using Adapt
using Random
using Zygote

const USE_METAL = Ref{Union{Nothing, Bool}}(nothing)

"""
    FluxMetalExt.FluxMetalDevice <: Flux.AbstractDevice

A type representing `device` objects for the `"Metal"` backend for Flux.
"""
Base.@kwdef struct FluxMetalDevice <: Flux.AbstractDevice
    deviceID::MTLDevice
end

(::FluxMetalDevice)(x) = gpu(FluxMetalAdaptor(), x)
Flux._isavailable(::FluxMetalDevice) = true
Flux._isfunctional(::FluxMetalDevice) = Metal.functional()
Flux._get_device_name(::FluxMetalDevice) = "Metal"

function check_use_metal()
    isnothing(USE_METAL[]) || return

    USE_METAL[] = Metal.functional()
    if !USE_METAL[]
        @info """
        The Metal function is being called but Metal.jl is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return
end
ChainRulesCore.@non_differentiable check_use_metal()

include("functor.jl")

function __init__()
    Flux.METAL_LOADED[] = true
    Flux.DEVICES[Flux.GPU_BACKEND_ORDER["Meta"]] = FluxMetalDevice(Metal.current_device())
end

end
