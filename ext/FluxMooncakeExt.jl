module FluxMooncakeExt

using ADTypes: AutoMooncake
using Mooncake: Mooncake
import Flux

function Flux.gradient(f::F, adtype::AutoMooncake, args::Vararg{Any,N}) where {F,N}
    return Flux.withgradient(f, adtype, args...)[2]
end

function Flux.withgradient(f::F, adtype::AutoMooncake, args::Vararg{Any,N}) where {F,N}
    config = Mooncake.Config(friendly_tangents=true)
    cache = Mooncake.prepare_pullback_cache(f, args...; config)
    # `prepare_pullback_cache` already runs the forward pass once to build the rule, and stores
    # a primal-typed buffer of the output `y = f(args...)`. We reuse it to learn the structure
    # of the output (without an extra forward pass) and to build the cotangent seed.
    yout = cache.y_cache
    seed = yout isa Union{Tuple, NamedTuple} ? _loss_seed(yout) : one(yout)
    # `value_and_pullback!!` does a single forward + reverse and returns the full output `y`,
    # so auxiliary outputs come for free in `val`.
    val, grads = Mooncake.value_and_pullback!!(cache, seed, f, args...)
    return (val=val, grad=grads[2:end])
end

# Auxiliary outputs: `f` returns a Tuple or NamedTuple whose first element is the scalar loss.
# We differentiate only the loss by seeding it with `one` and the auxiliary slots with zero
# cotangents, so the gradient flows through the loss alone while the whole output is returned.
_loss_seed(y::Tuple) = (one(first(y)), map(_zero_seed, Base.tail(y))...)
_loss_seed(y::NamedTuple{names}) where {names} =
    merge(NamedTuple{(names[1],)}((one(first(y)),)), map(_zero_seed, Base.tail(y)))

# A primal-typed zero cotangent. Numeric leaves are zeroed; non-differentiable leaves (e.g.
# strings) are passed through, since Mooncake maps them to `NoTangent` regardless of value.
_zero_seed(x::Number) = zero(x)
_zero_seed(x::AbstractArray{<:Number}) = zero(x)
_zero_seed(x::AbstractArray) = map(_zero_seed, x)
_zero_seed(x::Tuple) = map(_zero_seed, x)
_zero_seed(x::NamedTuple) = map(_zero_seed, x)
_zero_seed(x) = x

end # module
