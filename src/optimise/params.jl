using DataFlow: OSet

children(x) = ()

params(ps, m) = foreach(m -> params(ps, m), children(m))

function params(m)
  ps = OSet()
  params(ps, m)
  return collect(ps)
end

struct Param{T}
  x::T
  Δ::T
end

convert(::Type{Param}, x::AbstractArray) = Param(x, zeros(x))
