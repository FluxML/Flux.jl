module CUDAint

using ..CUDA

import ..Flux: Flux
using ChainRulesCore
import NNlib, NNlibCUDA

include("cudnn.jl")

end
