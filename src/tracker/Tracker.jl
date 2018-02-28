module Tracker

export TrackedArray, TrackedVector, TrackedMatrix, param, back!

tracker(x) = nothing

istracked(x) = tracker(x) ≠ nothing
isleaf(x) = !istracked(x) || isleaf(tracker(x))
data(x) = istracked(x) ? data(tracker(x)) : x
grad(x) = grad(tracker(x))

struct Call{F,As<:Tuple}
  func::F
  args::As
end

Call(f, args...) = Call{typeof(f),typeof(args)}(f, args)

@inline (c::Call)() = c.func(data.(c.args)...)

mutable struct Tracked{T}
  ref::UInt32
  f::Call
  isleaf::Bool
  data::T
  grad::T
  Tracked{T}(f::Call, data::T) where T = new(0, f, false, data)
  Tracked{T}(f::Call, data::T, grad::T) where T = new(0, f, false, data, grad)
  Tracked{T}(f::Call{Void}, data::T, grad::T) where T = new(0, f, true, data, grad)
end

Tracked(f::Call, x) = Tracked{typeof(x)}(f, x)
Tracked(f::Call, x, Δ) = Tracked{typeof(x)}(f, x, Δ)

track(f::Call, x) = Tracked(f, x)
track(f::Call) = track(f, f())
track(f, xs...) = track(Call(f, xs...))

istracked(x::Tracked) = true
isleaf(x::Tracked) = x.f == Call(nothing)
data(x::Tracked) = x.data
grad(x::Tracked) = x.grad

include("back.jl")
include("scalar.jl")
include("array.jl")
include("numeric.jl")

param(x::Number) = TrackedReal(float(x))
param(xs::AbstractArray) = TrackedArray(float.(xs))

import Adapt.adapt

adapt(T, xs::TrackedArray) = param(adapt(T, data(xs)))

end
