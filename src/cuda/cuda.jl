module CUDA

using ..CuArrays

if isdefined(CuArrays, :libcudnn_handle)
  handle() = CuArrays.libcudnn_handle[]
else
  handle() = CuArrays.CUDNN.handle()
end

end
