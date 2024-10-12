module FluxMetalExt

import Flux
import Flux: FluxCPUAdaptor, FluxMetalAdaptor, _metal, _isleaf, adapt_storage, fmap
import NNlib
using ChainRulesCore
using MLDataDevices: MLDataDevices
using Metal
using Adapt
using Random
using Zygote

const USE_METAL = Ref{Union{Nothing, Bool}}(nothing)

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
end

end
