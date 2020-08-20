module CUDAint

using ..CUDA

using CUDA: CUDNN
include("curnn.jl")
include("cudnn.jl")

end
