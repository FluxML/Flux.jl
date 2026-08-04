module FluxFiniteDifferencesExt

using Flux
using ADTypes: AutoFiniteDifferences
using FiniteDifferences

function Flux.gradient(f::F, adtype::AutoFiniteDifferences, x; autocast=nothing) where F
  g = Flux._autocast_closure(f, autocast)
  ps, re = Flux.destructure(x)
  gs = FiniteDifferences.grad(adtype.fdm, p -> g(re(p)...), ps)[1]
  return (re(gs),)
end

function Flux.gradient(f::F, adtype::AutoFiniteDifferences, x::Vararg{Any,N}; autocast=nothing) where {F, N}
  g = Flux._autocast_closure(f, autocast)
  ps, re = Flux.destructure(x)
  gs = FiniteDifferences.grad(adtype.fdm, p -> g(re(p)...), ps)[1]
  return re(gs)
end

function Flux.withgradient(f::F, adtype::AutoFiniteDifferences, x; autocast=nothing) where F
  g = Flux._autocast_closure(f, autocast)
  ps, re = Flux.destructure(x)
  y = g(re(ps)...)
  gs = FiniteDifferences.grad(adtype.fdm, p -> g(re(p)...), ps)[1]
  return y, (re(gs),)
end

function Flux.withgradient(f::F, adtype::AutoFiniteDifferences, x::Vararg{Any,N}; autocast=nothing) where {F, N}
  g = Flux._autocast_closure(f, autocast)
  ps, re = Flux.destructure(x)
  y = g(re(ps)...)
  gs = FiniteDifferences.grad(adtype.fdm, p -> g(re(p)...), ps)[1]
  return y, re(gs)
end

end # module
