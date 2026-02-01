module FluxMooncakeExt

using ADTypes: AutoMooncake
using Mooncake: Mooncake
import Flux

function Flux.gradient(f::F, adtype::AutoMooncake, args::Vararg{Any,N}) where {F,N}
    return Flux.withgradient(f, adtype, args...)[2]
end

function Flux.withgradient(f::F, adtype::AutoMooncake, args::Vararg{Any,N}) where {F,N}
    cache = Mooncake.prepare_gradient_cache(f, args...; config=Mooncake.Config(friendly_tangents=true))
    val, grads = Mooncake.value_and_gradient!!(cache, f, args...)
    return (val=val, grad=grads[2:end])
end 

end # module
