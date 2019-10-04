module CUDA

using ..CuArrays

if CuArrays.libcudnn !== nothing  # TODO: use CuArrays.has_cudnn()
  using CuArrays: CUDNN
  include("curnn.jl")
  include("cudnn.jl")
else
  @warn "CUDNN is not installed, some functionality will not be available."
end

end
