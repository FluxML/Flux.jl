module CUDAint

using ..CUDA

import ..Flux: Flux
import Zygote
using Zygote: @adjoint
import NNlib, NNlibCUDA

include("cudnn.jl")

end
