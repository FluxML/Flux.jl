module CUDA

using CuArrays

if CuArrays.cudnn_available()
    include("curnn.jl")
    include("cudnn.jl")
end

end
