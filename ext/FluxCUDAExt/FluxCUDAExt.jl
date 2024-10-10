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
include("utils.jl")

function __init__()
    Flux.CUDA_LOADED[] = true

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
