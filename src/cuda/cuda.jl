module CUDA

using ..CuArrays

if CuArrays.libcudnn != nothing
  include("cudnn.jl")
else
  @warn("CUDNN is not installed, some functionality will not be available.")
end

end
