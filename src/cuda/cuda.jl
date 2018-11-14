module CUDA

using ..CuArrays

if !applicable(CuArray{UInt8}, undef, 1)
  (T::Type{<:CuArray})(::UndefInitializer, sz...) = T(sz...)
end

if CuArrays.libcudnn != nothing
  include("cudnn.jl")
else
  @warn("CUDNN is not installed, some functionality will not be available.")
end

end
