using Test

using CUDA
CUDA.allowscalar(false)

using Flux, FluxCUDA
Flux.default_gpu_converter[] = cu

using Zygote
using Zygote: pullback

include("test_utils.jl")
include("cuda.jl")
include("losses.jl")
include("layers.jl")

if CUDA.has_cudnn()
  @info "Testing Flux/CUDNN"
  include("cudnn.jl")
  include("curnn.jl")
else
  @warn "CUDNN unavailable, not testing GPU DNN support"
end
