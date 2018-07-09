module Tracker

using MacroTools
using MacroTools: @q, @forward

import Base: ==

export TrackedArray, TrackedVector, TrackedMatrix, param, back!

tracker(x) = nothing

istracked(x) = tracker(x) ≠ nothing
isleaf(x) = !istracked(x) || isleaf(tracker(x))
data(x) = istracked(x) ? data(tracker(x)) : x
grad(x) = grad(tracker(x))
grad(::Void) = nothing

struct Call{F,As<:Tuple}
  func::F
  args::As
end

Call(f, args) = Call{typeof(f),typeof(args)}(f, args)
Call() = Call(nothing, ())

# When deserialising, the object_id changes
a::Call == b::Call = a.func == b.func && a.args == b.args

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

istracked(x::Tracked) = true
isleaf(x::Tracked) = x.f == Call()
data(x::Tracked) = x.data
grad(x::Tracked) = x.grad

track(f::Call, x) = Tracked(f, x)
track(f::Call) = track(f, f())

function _forward end

function track(f, xs...; kw...)
  y, back = _forward(f, data.(xs)...; kw...)
  track(Call(back, xs), y)
end

macro grad(ex)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {T__} = body_)) || error("Need a function definition")
  T == nothing && (T = [])
  insert!(args, 1+isexpr(args[1], :parameters) , :(::typeof($name)))
  @q(Tracker._forward($(args...)) where $(T...) = $body) |> esc
end

function update!(x, Δ)
  tracker(x).data += Δ
  tracker(x).grad .= 0
  return x
end

include("idset.jl")
include("back.jl")
include("scalar.jl")
include("array.jl")
include("numeric.jl")

"""
    hook(f, x) -> x′

Hook into gradient backpropagation. `x` is unmodified, but when backpropagating
`f` will be applied to the incoming gradient. For example, `hook(-, x)` will reverse
the sign of the gradient applied to `x`.
"""
hook(f, x) = istracked(x) ? track(hook, f, x) : x
@grad hook(f, x) = x, Δ -> (nothing, f(Δ))

param(x::Number) = TrackedReal(float(x))
param(xs::AbstractArray) = TrackedArray(float.(xs))

import NNlib.cudata
import Adapt.adapt

cudata(x::TrackedArray) = data(x)
adapt(T, xs::TrackedArray) = param(adapt(T, data(xs)))

end
