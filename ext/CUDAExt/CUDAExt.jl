module CUDAExt

using CUDA
import NNlib, NNlibCUDA

using Flux
import Flux: adapt_storage, _gpu, FluxCPUAdaptor, _isleaf, dropout_mask, _dropout_mask

using Adapt
using ChainRulesCore
using Random
using Zygote

const use_cuda = Ref{Union{Nothing,Bool}}(nothing)

include("utils.jl")
include("functor.jl")

include("layers/normalise.jl")

include("cudnn.jl")

function __init__()
    Flux.cuda_loaded[] = true
end

end # module
