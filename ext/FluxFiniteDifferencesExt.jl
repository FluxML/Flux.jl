module FluxFiniteDifferencesExt

using Flux
using ADTypes: AutoFiniteDifferences
using FiniteDifferences

function Flux.gradient(f::F, adtype::AutoFiniteDifferences, x) where F
    ps, re = Flux.destructure(x)
    gs = FiniteDifferences.grad(adtype.fdm, p -> f(re(p)...), ps)[1]
    return (re(gs),)
end

function Flux.gradient(f::F, adtype::AutoFiniteDifferences, x::Vararg{Any,N}) where {F, N}
    ps, re = Flux.destructure(x)
    gs = FiniteDifferences.grad(adtype.fdm, p -> f(re(p)...), ps)[1]
    return re(gs)
end

function Flux.withgradient(f::F, adtype::AutoFiniteDifferences, x) where F
    ps, re = Flux.destructure(x)
    y = f(re(ps)...)
    gs = FiniteDifferences.grad(adtype.fdm, p -> f(re(p)...), ps)[1]
    return y, (re(gs),)
end

function Flux.withgradient(f::F, adtype::AutoFiniteDifferences, x::Vararg{Any,N}) where {F, N}
    ps, re = Flux.destructure(x)
    y = f(re(ps)...)
    gs = FiniteDifferences.grad(adtype.fdm, p -> f(re(p)...), ps)[1]
    return y, re(gs)
end

end # module
