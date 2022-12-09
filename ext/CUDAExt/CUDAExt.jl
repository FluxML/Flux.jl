module CUDAExt

using CUDA
import NNlib, NNlibCUDA

using Flux
import Flux: adapt_storage, _gpu, FluxCPUAdaptor, _isleaf

using Adapt
using ChainRulesCore
using Random
using Zygote

const use_cuda = Ref{Union{Nothing,Bool}}(nothing)

include("utils.jl")
include("functor.jl")

include("layers/normalise.jl")

include("cudnn.jl")

end # module
