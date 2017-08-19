module Tracker

data(x) = x

struct Call{F,As<:Tuple}
  func::F
  args::As
end

Call(f, args...) = Call{typeof(f),typeof(args)}(f, args)

(c::Call)() = c.func(data.(c.args)...)

back!(c::Call, Δ) = back!(c.func, Δ, c.args...)

back!(f, Δ) = nothing

struct Var{T,N,A} <: AbstractArray{T,N}
  f::Call
  x::A
  Δ::A
end

ScalarVar{T,A} = Var{T,0,A}
VectorVar{T,A} = Var{T,1,A}
MatrixVar{T,A} = Var{T,2,A}

Var(c::Call, x::A, Δ::A) where A <: AbstractArray =
  Var{eltype(A),ndims(A),A}(c, x, Δ)

Var(c::Call, x::AbstractArray) = Var(c, x, zeros(x))

Var(c::Call) = Var(c, c())

Var(x::AbstractArray) = Var(Call(nothing), x)

data(x::Var) = x.x
grad(x::Var) = x.Δ

function back!(x::Var, Δ)
  x.Δ .+= Δ
  back!(x.f, Δ)
end

for f in :[Base.size, Base.ndims, Base.similar].args
  @eval @inline $f(x::Var, a...) = $f(data(x), a...)
end

function Base.showarray(io::IO, X::Var, repr::Bool = true; header = true)
  if repr
    print(io, "Var(")
    Base.showarray(io, data(X), true)
    print(io, ")")
  else
    println(io, summary(X), ":")
    Base.showarray(io, data(X), false, header = false)
  end
end

end
