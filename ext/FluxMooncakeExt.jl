module FluxMooncakeExt

import Flux: _grad_unwrap
using ADTypes: AutoMooncake
using Mooncake: Tangent, MutableTangent
using Functors: fmap

_grad_unwrap(adtype::AutoMooncake, x) = fmap(y -> _grad_unwrap(adtype, y), x, exclude= y -> y isa Union{Tangent, MutableTangent})
_grad_unwrap(adtype::AutoMooncake, x::Tangent) = _grad_unwrap(adtype, x.fields)
_grad_unwrap(adtype::AutoMooncake, x::MutableTangent) = _grad_unwrap(adtype, x.fields)

end # module
