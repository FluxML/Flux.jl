import Adapt: adapt

children(x) = ()
mapchildren(f, x) = x

children(x::Tuple) = x
mapchildren(f, x::Tuple) = map(f, x)

function treelike(T, fs = fieldnames(T))
  @eval begin
    children(x::$T) = ($([:(x.$f) for f in fs]...),)
    mapchildren(f, x::$T) = $T(f.(children(x))...)
    adapt(T, x::$T) = mapchildren(x -> adapt(T, x), x)
  end
end

isleaf(x) = isempty(children(x))

function mapleaves(f, x; cache = ObjectIdDict())
  haskey(cache, x) && return cache[x]
  cache[x] = isleaf(x) ? f(x) : mapchildren(x -> mapleaves(f, x, cache = cache), x)
end

export mapparams
@deprecate mapparams(f, x) mapleaves(f, x)

using DataFlow: OSet

function prefor(f, x; seen = OSet())
  x ∈ seen && return
  f(x)
  foreach(x -> prefor(f, x, seen = seen), children(x))
  return
end

using Flux.Tracker: istracked

function params(m)
  ps = []
  prefor(p -> istracked(p) && push!(ps, p), m)
  return ps
end

params(m...) = params(m)
