type Param{T}
  x::T
  Δx::T
end

param(x) = Param(x, zero(x))

state(p::Param) = p.x

function accumulate!(p::Param, Δ)
  p.Δx .+= Δ
  return p
end

function update!(p::Param, η)
  p.x .+= p.Δx .* η
  return p
end

state(x) = x
accumulate!(x, Δ) = x
