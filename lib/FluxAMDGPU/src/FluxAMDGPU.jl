module FluxAMDGPU

using Flux
using AMDGPU

### onehot

import Flux: OneHotArray, OneHotLike, _onehot_bool_type

_onehot_bool_type(x::OneHotLike{<:Any, <:Any, <:Any, N, <:ROCArray}) where N = ROCArray{Bool, N}
Base.BroadcastStyle(::Type{<:OneHotArray{<:Any, <:Any, <:Any, N, <:ROCArray}}) where N = AMDGPU.ROCArrayStyle{N}()

end # module
