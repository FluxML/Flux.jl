module CUDAint

import ..Flux: Flux
using Flux: FluxCPUAdaptor
using ChainRulesCore
using CUDA
using Random
import NNlib, NNlibCUDA
using Functors
using Adapt
import Adapt: adapt_storage
import Zygote

include("utils.jl")
include("functor.jl")
include("cudnn.jl")

end # module
