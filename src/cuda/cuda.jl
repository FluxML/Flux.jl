module CUDAint

using ..CUDA
using CUDA: CUDNN

import ..Flux: Flux
import Zygote
using Zygote: @adjoint

include("cudnn.jl")

end
