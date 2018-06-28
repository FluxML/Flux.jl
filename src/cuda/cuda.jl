module CUDA

using CuArrays

if CuArrays.cudnn_available()
    CuParam{T,N} = Union{CuArray{T,N},TrackedArray{T,N,CuArray{T,N}}}
    include("curnn.jl")
    include("cudnn.jl")
end

end
