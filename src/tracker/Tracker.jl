module Tracker

using MacroTools
using MacroTools: @q, @forward

import Base: ==

export TrackedArray, TrackedVector, TrackedMatrix, Params, param, back!

tracker(x) = nothing

istracked(x) = tracker(x) ≠ nothing
isleaf(x) = !istracked(x) || isleaf(tracker(x))
grad(x) = grad(tracker(x))
grad(::Nothing) = nothing
data(x) = x

struct Call{F,As<:Tuple}
  func::F
  args::As
end

Call(f::F, args::T) where {F,T} = Call{F,T}(f, args)
Call() = Call(nothing, ())

# When deserialising, the object_id changes
a::Call == b::Call = a.func == b.func && a.args == b.args

@inline (c::Call)() = c.func(data.(c.args)...)

mutable struct Tracked{T}
  ref::UInt32
  f::Call
  isleaf::Bool
  grad::T
  Tracked{T}(f::Call) where T = new(0, f, false)
  Tracked{T}(f::Call, grad::T) where T = new(0, f, false, grad)
  Tracked{T}(f::Call{Nothing}, grad::T) where T = new(0, f, true, grad)
end

istracked(x::Tracked) = true
isleaf(x::Tracked) = x.f == Call()
grad(x::Tracked) = x.grad

track(f::Call, x) = Tracked{typeof(x)}(f)

function _forward end

function track(f::F, xs...) where F
  y, back = _forward(f, xs...)
  ts = map(tracker, xs)
  c = Call(back, ts)
  track(c, y)
end

function track_kw(f::F, xs...; kw...) where F
  y, back = _forward(f, xs...; kw...)
  track(Call(back, tracker.(xs)), y)
end

macro grad(ex)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {T__} = body_)) || error("Need a function definition")
  T == nothing && (T = [])
  isexpr(name, :(::)) || (name = :(::typeof($name)))
  insert!(args, 1+isexpr(args[1], :parameters) , name)
  @q(Tracker._forward($(args...)) where $(T...) = $body) |> esc
end

function update!(x, Δ)
  x.data .+= data(Δ)
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

"""
    checkpoint(f, args...)

Behaves like `f(args...)`, but avoids storing the intermediate values needed for
calculating gradients. Instead, `f(args...)` will be called again during the
backward pass. This can be used to save memory in larger models.
"""
checkpoint(f, args...) = track(checkpoint, f, args...)

@grad function checkpoint(f, args...)
  data(f(args...)), function (Δ)
    y, back = forward(f, args...)
    (nothing, back(Δ)...)
  end
end

nobacksies(f, x) = track(nobacksies, f, x)
nobacksies(f, xs::Tuple) = map(x -> nobacksies(f, x), xs)
@grad nobacksies(f, x) = data(x), Δ -> error("Nested AD not defined for $f")

param(x::Number) = TrackedReal(float(x))
param(xs::AbstractArray) = TrackedArray(float.(xs))

@grad identity(x) = data(x), Δ -> (Δ,)
param(x::TrackedReal) = track(identity, x)
param(x::TrackedArray) = track(identity, x)

import NNlib.cudata
import Adapt.adapt

cudata(x::TrackedArray) = data(x)
adapt(T, xs::TrackedArray) = param(adapt(T, data(xs)))

end
