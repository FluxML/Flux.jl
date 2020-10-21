module CUDAint

using ..CUDA

using CUDA: CUDNN

import ..Flux: Flux
import Zygote
using Zygote: @adjoint

# include("curnn.jl")
include("cudnn.jl")

end
