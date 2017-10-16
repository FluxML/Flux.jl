children(x) = ()
mapchildren(f, x) = x

function treelike(T, fs = fieldnames(T))
  @eval begin
    children(x::$T) = ($([:(x.$f) for f in fs]...),)
    mapchildren(f, x::$T) = $T(f.(children(x))...)
  end
end

# TODO: prewalk/postwalk with correct caching
# This is only correct in general for idempotent functions

isleaf(x) = isempty(children(x))

fmap(f, x) = isleaf(x) ? f(x) : mapchildren(x -> fmap(f, x), x)
ffor(f, x) = isleaf(x) ? f(x) : foreach(x -> ffor(f, x), children(x))

using DataFlow: OSet

function params(m)
  ps, seen = [], OSet()
  ffor(m) do p
    p isa TrackedArray && p ∉ seen &&
      (push!(ps, p); push!(seen, p))
  end
  return ps
end
