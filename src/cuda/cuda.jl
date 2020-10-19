module CUDAint

using ..CUDA

using CUDA: CUDNN
# include("curnn.jl")
include("curnn_jdb_v1.jl")
include("cudnn.jl")

end
