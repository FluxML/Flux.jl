module CUDA

using ..CuArrays

CuArrays.libcudnn != nothing && include("cudnn.jl")

end
