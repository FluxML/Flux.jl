module CUDA

using ..CuArrays

using CuArrays: CUDNN
include("curnn.jl")
include("cudnn.jl")

end
