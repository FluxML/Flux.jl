module CUDA

using ..CuArrays

if has_cudnn()
  include("curnn.jl")
  include("cudnn.jl")
else
  @warn "CUDNN is not installed, some functionality will not be available."
end

end
