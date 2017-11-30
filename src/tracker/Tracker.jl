module Tracker

export TrackedArray, TrackedVector, TrackedMatrix, param, back!

data(x) = x
istracked(x) = false

struct Call{F,As<:Tuple}
  func::F
  args::As
end

Call(f, args...) = Call{typeof(f),typeof(args)}(f, args)

(c::Call)() = c.func(data.(c.args)...)

mutable struct TrackedArray{T,N,A} <: AbstractArray{T,N}
  ref::UInt32
  f::Call
  data::A
  grad::A
  TrackedArray{T,N,A}(f::Call, data::A) where {T,N,A} = new(0, f, data)
  TrackedArray{T,N,A}(f::Call, data::A, grad::A) where {T,N,A} = new(0, f, data, grad)
end

TrackedScalar{T,A} = TrackedArray{T,0,A}
TrackedVector{T,A} = TrackedArray{T,1,A}
TrackedMatrix{T,A} = TrackedArray{T,2,A}
TrackedVecOrMat{T,A} = Union{TrackedVector{T,A},TrackedMatrix{T,A}}

TrackedArray(c::Call, x::A) where A <: AbstractArray =
  TrackedArray{eltype(A),ndims(A),A}(c, x)

TrackedArray(c::Call, x::A, Δ::A) where A <: AbstractArray =
  TrackedArray{eltype(A),ndims(A),A}(c, x, Δ)

TrackedArray(c::Call) = TrackedArray(c, c())

TrackedArray(x::AbstractArray) = TrackedArray(Call(nothing), x, zeros(x))

isleaf(x::TrackedArray) = x.f == Call(nothing)

param(xs) = TrackedArray(map(x -> AbstractFloat(x), xs))
param(xs::Real) = param(fill(xs))

istracked(x::TrackedArray) = true
data(x::TrackedArray) = x.data
grad(x::TrackedArray) = x.grad

# Fallthrough methods

for f in :[Base.size, Base.ndims].args
  @eval @inline $f(x::TrackedArray, a...) = $f(data(x), a...)
end

Base.similar(x::TrackedArray, dims::Union{AbstractUnitRange,Integer}...) =
  similar(data(x), dims...)

Base.similar(x::TrackedArray, T::Type) = similar(data(x), T)

value(x) = x
value(x::TrackedArray) = data(x)
value(x::TrackedScalar) = data(x)[]

Base.:(==)(x::TrackedArray, y) = value(x) == y
Base.:(==)(y, x::TrackedArray) = y == value(x)
Base.:(==)(x::TrackedArray, y::TrackedArray) = value(x) == value(x)

Base.isless(x::TrackedScalar, y) = isless(value(x), y)
Base.isless(x, y::TrackedScalar) = isless(x, value(y))
Base.isless(x::TrackedScalar, y::TrackedScalar) = isless(value(x), value(y))

Base.show(io::IO, ::Type{TrackedArray{T,N,A}}) where {T,N,A<:AbstractArray{T,N}} =
  print(io, "TrackedArray{…,$A}")

function Base.showarray(io::IO, X::TrackedArray, repr::Bool = true; header = true)
  if repr
    print(io, "param(")
    Base.showarray(io, data(X), true)
    print(io, ")")
  else
    header && print(io, "Tracked ")
    Base.showarray(io, data(X), false, header = header)
  end
end

Base.setindex!(xs::TrackedArray, v, i...) =
  error("Can't differentiate `setindex!`")

include("back.jl")
include("lib.jl")
include("numeric.jl")

import NNlib.adapt

adapt(T, xs::TrackedArray) = TrackedArray(xs.f, adapt(T, xs.data), adapt(T, xs.grad))

end
