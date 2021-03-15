module CUDAint

using ..CUDA
# using CUDA: CUDNN

import ..Flux: Flux
import Zygote
using Zygote: @adjoint
import NNlib, NNlibCUDA

include("cudnn.jl")

end
