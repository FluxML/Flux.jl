module FluxFiniteDifferencesExt

using Flux
using ADTypes: AutoFiniteDifferences
using FiniteDifferences

function Flux.gradient(f::F, adtype::AutoFiniteDifferences, x; autocast=nothing) where F
  Flux._with_autocast(autocast) do
    ps, re = Flux.destructure(x)
    gs = FiniteDifferences.grad(adtype.fdm, p -> f(re(p)...), ps)[1]
    return (re(gs),)
  end
end

function Flux.gradient(f::F, adtype::AutoFiniteDifferences, x::Vararg{Any,N}; autocast=nothing) where {F, N}
  Flux._with_autocast(autocast) do
    ps, re = Flux.destructure(x)
    gs = FiniteDifferences.grad(adtype.fdm, p -> f(re(p)...), ps)[1]
    return re(gs)
  end
end

function Flux.withgradient(f::F, adtype::AutoFiniteDifferences, x; autocast=nothing) where F
  Flux._with_autocast(autocast) do
    ps, re = Flux.destructure(x)
    y = f(re(ps)...)
    gs = FiniteDifferences.grad(adtype.fdm, p -> f(re(p)...), ps)[1]
    return y, (re(gs),)
  end
end

function Flux.withgradient(f::F, adtype::AutoFiniteDifferences, x::Vararg{Any,N}; autocast=nothing) where {F, N}
  Flux._with_autocast(autocast) do
    ps, re = Flux.destructure(x)
    y = f(re(ps)...)
    gs = FiniteDifferences.grad(adtype.fdm, p -> f(re(p)...), ps)[1]
    return y, re(gs)
  end
end

end # module
