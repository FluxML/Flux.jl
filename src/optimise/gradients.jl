struct ZygoteImplicitBackend{T} <: AD.AbstractReverseMode
    core_backend::T
end
ZygoteImplicitBackend() = ZygoteImplicitBackend(AD.ZygoteBackend())

AD.@primitive pullback_function(ad::ZygoteImplicitBackend, f, x::Zygote.Params) =
    AD.pullback_function(ad.core_backend, f, x)

# this is a hack to get around
# https://github.com/JuliaDiff/AbstractDifferentiation.jl/issues/63#issuecomment-1225959150
AD.gradient(::ZygoteImplicitBackend, f, x::Zygote.Params) = Zygote.gradient(f, x)
AD.value_and_gradient(::ZygoteImplicitBackend, f, x::Zygote.Params) =
    Zygote.withgradient(f, x)

struct ZygoteExplicitBackend{T} <: AD.AbstractReverseMode
    core_backend::T
end
ZygoteExplicitBackend() = ZygoteExplicitBackend(AD.ZygoteBackend())

AD.@primitive pullback_function(ad::ZygoteExplicitBackend, f, xs...) =
    AD.pullback_function(ad.core_backend, f, xs...)

# this is a hack to get around
# https://github.com/JuliaDiff/AbstractDifferentiation.jl/issues/63#issuecomment-1225959150
AD.gradient(::ZygoteExplicitBackend, f, xs...) = Zygote.gradient(f, xs...)
AD.value_and_gradient(::ZygoteExplicitBackend, f, xs...) =
    Zygote.withgradient(f, xs...)
