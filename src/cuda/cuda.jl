module CUDA

using ..CuArrays

if CuArrays.has_cudnn()
  using CuArrays: CUDNN
  include("curnn.jl")
  include("cudnn.jl")
else
  @warn "CUDNN is not installed, some functionality will not be available."
end

end
