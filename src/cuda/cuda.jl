module CUDA

using ..CuArrays

CuArrays.cudnn_available() && include("cudnn.jl")

end
