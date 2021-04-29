module FluxCUDA

using Flux, CUDA
using NNlib, NNlibCUDA
using Zygote
using Zygote: @adjoint

include("onehot.jl")
include("ctc.jl")
include("cudnn.jl")


end # module
