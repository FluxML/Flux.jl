module FluxAMDGPU

using Flux
using AMDGPU

### onehot

using Flux: OneHotArray, OneHotLike

Base.BroadcastStyle(::Type{<:OneHotArray{<:Any, <:Any, <:Any, N, <:ROCArray}}) where N =
    AMDGPU.ROCArrayStyle{N}()

end # module
