module CUDAint

using ..CUDA

import ..Flux: Flux
using Zygote: Zygote
using Zygote: @adjoint
using NNlib: NNlib
using NNlibCUDA: NNlibCUDA

include("cudnn.jl")

end
