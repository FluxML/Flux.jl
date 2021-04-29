import Flux: OneHotArray, OneHotLike, _onehot_bool_type

_onehot_bool_type(x::OneHotLike{<:Any, <:Any, <:Any, N, <:CuArray}) where N = CuArray{Bool, N}

Base.BroadcastStyle(::Type{<:OneHotArray{<: Any, <: Any, <: Any, N, <: CuArray}}) where N = CUDA.CuArrayStyle{N}()
