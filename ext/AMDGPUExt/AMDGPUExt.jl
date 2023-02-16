module AMDGPUExt

import ChainRulesCore
import ChainRulesCore: NoTangent
import Flux
import Flux: FluxCPUAdaptor, FluxAMDAdaptor, _amd, _isleaf, adapt_storage, fmap

using AMDGPU
using Adapt
using Random
using Zygote

const MIOPENFloat = AMDGPU.MIOpen.MIOPENFloat
const USE_AMDGPU = Ref{Union{Nothing, Bool}}(nothing)

function check_use_amdgpu()
    isnothing(USE_AMDGPU[]) || return

    USE_AMDGPU[] = AMDGPU.functional()
    if USE_AMDGPU[]
        if !AMDGPU.functional(:MIOpen)
            @warn "MIOpen is not functional in AMDGPU.jl, some functionality will not be available."
        end
    else
        @info """
        The AMDGPU function is being called but the AMDGPU is not functional.
        Defaulting back to the CPU. (No action is required if you want to run on the CPU).
        """ maxlog=1
    end
    return
end
ChainRulesCore.@non_differentiable check_use_amdgpu()

include("functor.jl")

function __init__()
    Flux.AMDGPU_LOADED[] = true
end

# TODO
# fail early if input to the model is not on the device (e.g. on the host)
# otherwise we get very cryptic errors & segfaults at the rocBLAS level

end
