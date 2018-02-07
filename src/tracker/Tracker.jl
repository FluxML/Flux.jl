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

(c::Call)() = c.func(data.(c.args)...)

mutable struct Tracked{T}
  ref::UInt32
  f::Call
  data::T
  grad::T
  Tracked{T}(f::Call, data::T) where T = new(0, f, data)
  Tracked{T}(f::Call, data::T, grad::T) where T = new(0, f, data, grad)
end

istracked(x::Tracked) = true
isleaf(x::Tracked) = x.f == Call(nothing)
data(x::Tracked) = x.data
grad(x::Tracked) = x.grad

include("back.jl")
include("array.jl")
include("numeric.jl")

param(x::Number) = TrackedArray(fill(0))
Base.isless(x::TrackedScalar, y) = isless(x.data[], y)
Base.isless(x, y::TrackedScalar) = isless(x, y.data[])
Base.isless(x::TrackedScalar, y::TrackedScalar) = isless(x.data[], y.data[])
back!(x::TrackedScalar) = back!(x, 1)

param(xs::AbstractArray) = TrackedArray(map(x -> AbstractFloat(x), xs))

using DataFlow
using DataFlow: inputnode, constant

vcall(f, args...) = vertex(DataFlow.Call(), constant(f), args...)
vcall(f::Broadcasted, args...) = vcall(broadcast, constant(f.f), args...)

function _graph(x::TrackedArray, inputs::TrackedArray...; cache = ObjectIdDict())
  haskey(cache, x) && return cache[x]
  i = findfirst(inputs, x)
  cache[x] =
    i > 0 ? inputnode(i) :
    isleaf(x) ? constant(x) :
    vcall(x.f.func, map(x -> _graph(x, inputs...; cache = cache), x.f.args)...)
end

_graph(x, inputs::TrackedArray...; cache = ObjectIdDict()) = constant(x)

function graph(f, args...)
  inputs = param.(args)
  _graph(f(inputs...), inputs...)
end

import Adapt.adapt

adapt(T, xs::TrackedArray) = TrackedArray(xs.f, adapt(T, xs.data), adapt(T, xs.grad))

end
