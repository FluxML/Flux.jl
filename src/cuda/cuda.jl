module CUDAint

using ..CUDA

import ..Flux: Flux
import Zygote
using Zygote: @adjoint
using NNlib: NNlib
using NNlibCUDA: NNlibCUDA

include("cudnn.jl")

end
