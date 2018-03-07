module CUDA

using CuArrays

CuArrays.cudnn_available() && include("cudnn.jl")

import ..Flux.JIT: Shape, restructure

function restructure(sh::Shape{T}, buf::CuVector{UInt8}) where T
  buf = buf[1:sizeof(sh)]
  reshape(reinterpret(T, buf), size(sh))
end

end
