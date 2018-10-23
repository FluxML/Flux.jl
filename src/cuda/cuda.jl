module CUDA

using ..CuArrays

if CuArrays.libcudnn != nothing
    include("curnn.jl")
    include("cudnn.jl")
end

end
