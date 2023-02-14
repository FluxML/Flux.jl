module AMDGPUExt

using AMDGPU
using Adapt
using Random
using Zygote
import ChainRulesCore
import Functors: fmap
import Flux
import Flux: FluxCPUAdaptor, adapt_storage, _isleaf, _amd

const use_amdgpu = Ref{Bool}(false)

include("functor.jl")

function __init__()
    Flux.amdgpu_loaded[] = true
end

end
