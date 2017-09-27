children(x) = ()
mapchildren(f, x) = x

function treelike(T, fs = fieldnames(T))
  @eval begin
    children(x::$T) = ($([:(x.$f) for f in fs]...),)
    mapchildren(f, x::$T) = $T(f.(children(x))...)
  end
end

using DataFlow: OSet

params(ps, p::AbstractArray) = push!(ps, p)
params(ps, m) = foreach(m -> params(ps, m), children(m))

function params(m)
  ps = OSet()
  params(ps, m)
  return collect(ps)
end
