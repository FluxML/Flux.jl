using Flux, Test, CUDA

@info "Testing GPU Support"
CUDA.allowscalar(false)

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