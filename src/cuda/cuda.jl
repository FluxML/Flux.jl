module CUDA

using CuArrays

if CuArrays.cudnn_available()
    include("cudnn.jl")
    include("curnn.jl")
end

end
