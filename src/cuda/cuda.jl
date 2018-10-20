module CUDA

using ..CuArrays

isdefined(CuArrays, :CUDNN) && include("cudnn.jl")

end
