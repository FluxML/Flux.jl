module Tracker

using Base: RefValue

export TrackedArray, param, back!

data(x) = x
istracked(x) = false

struct Call{F,As<:Tuple}
  func::F
  args::As
end

Call(f, args...) = Call{typeof(f),typeof(args)}(f, args)

(c::Call)() = c.func(data.(c.args)...)

struct TrackedArray{T,N,A} <: AbstractArray{T,N}
  ref::RefValue{UInt32}
  f::Call
  data::A
  grad::RefValue{A}
end

TrackedScalar{T,A} = TrackedArray{T,0,A}
TrackedVector{T,A} = TrackedArray{T,1,A}
TrackedMatrix{T,A} = TrackedArray{T,2,A}
TrackedVecOrMat{T,A} = Union{TrackedVector{T,A},TrackedMatrix{T,A}}

TrackedArray(c::Call, x::A, Δ::Ref{A}) where A <: AbstractArray =
  TrackedArray{eltype(A),ndims(A),A}(Ref(UInt32(0)), c, x, Δ)

TrackedArray(c::Call, x::AbstractArray) = TrackedArray(c, x, RefValue{typeof(x)}())

TrackedArray(c::Call) = TrackedArray(c, c())

TrackedArray(x::AbstractArray) = TrackedArray(Call(nothing), x, RefValue(zeros(x)))

param(xs) = TrackedArray(AbstractFloat.(xs))
istracked(x::TrackedArray) = true
data(x::TrackedArray) = x.data
grad(x::TrackedArray) = x.grad[]

# Fallthrough methods

for f in :[Base.size, Base.ndims].args
  @eval @inline $f(x::TrackedArray, a...) = $f(data(x), a...)
end

Base.similar(x::TrackedArray, dims::Union{AbstractUnitRange,Integer}...) =
  similar(data(x), dims...)

Base.similar(x::TrackedArray, T::Type) = similar(data(x), T)

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

include("back.jl")
include("lib.jl")
include("numeric.jl")

import NNlib.adapt

adapt(T, xs::TrackedArray) =
  TrackedArray(xs.f, adapt(T, xs.data),
               RefValue(adapt(T, grad(xs))))

end
