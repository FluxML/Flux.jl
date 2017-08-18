struct Call{F,As<:Tuple}
  func::F
  args::As
end

Call(f, args...) = Call{typeof(f),typeof(args)}(f, args)

back!(c::Call, Δ) = back!(c.func, Δ, c.args...)
back!(::Void, Δ) = nothing

mutable struct Var{T}
  f::Call
  x::T
  Δ::T
end

Var(x::T, Δ::T) where {T} = Var(Call(nothing), x, Δ)
Var(x::AbstractArray) = Var(x, zeros(x))
Var(x::Number) = Var(x, zero(x))

function back!(x::Var, Δ)
  x.Δ .+= Δ
  back!(x.f, Δ)
end
