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

mapparams(f, x::AbstractArray) = f(x)
mapparams(f, x) = mapchildren(x -> mapparams(f, x), x)

forparams(f, x) = (mapparams(x -> (f(x); x), x); return)

using DataFlow: OSet

function params(m)
  ps = OSet()
  forparams(p -> push!(ps, p), m)
  return collect(ps)
end
