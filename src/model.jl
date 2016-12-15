export Model, back!, update!, param

# Basic model API

"""
    (m::Model)(X...) => Y

A "model" is a function with state. For example, a logistic regression is the
function

    x -> σ(x * W + b)

where `W` and `b` are a trainable matrix and vector of weights repectively. The
`Model` abstract type is used loosely; in general the concept of a model is
closer to a protocol, and models don't need to inherit from this type. Normal
Julia functions are models with 0 parameters, for example.
"""
abstract Model

"""
    back!(m::Model, ΔY, X...) => ΔX

Backpropagate the gradient `ΔY` through the model `m`, accumulating the
gradients of any parameters. Returns the gradient of the input `X`. Gradients
may be arrays or tuples of arrays (for multiple inputs/outputs).
"""
back!(m::Model, Δ, xs...) = error("Backprop not implemented for $(typeof(m))")

"""
    update!(m::Model, η) => m

Update the parameters of the model `m` using the accumulated gradients from
`back!`, using the learning rate `η`.
"""
update!(m, η) = m

"""
    graph(m::Model) => ::IVertex{Any} | nothing

Returns the graph representation of the model, if any. Most models are built
from lower-level components and can simply implement this method to get most of
Flux's functionality. If this method isn't available, functionality like
backpropagation or conversion for backend must be implemented on a case-by-case
basis. Alternatively, one can implement this method and override individual
methods as necessary.
"""
graph(m) = nothing

# Model parameters

"""
A `Param` object stores a parameter array along with an accumulated delta to
that array. When converting to backends like TensorFlow, identical `Param`s will
result in identical variable objects, making model reuse trivial.
"""
type Param{T}
  x::T
  Δx::T
end

"""
    param(x::T) => ::Param{T}

Convenience method for creating a `Param` object for a given array.
"""
param(x) = Param(x, zero(x))

state(p::Param) = p.x

"""
    accumulate!(p::Param, Δ) => p

Accumulates the update `Δ` on `p`. The value of `p` won't change until
`update!`.
"""
function accumulate!(p::Param, Δ)
  p.Δx += Δ
  return p
end

"""
    update!(p::Param)

Apply the accumulated updates to the value of the parameter.
"""
function update!(p::Param, η)
  p.x .-= p.Δx .* η
  p.Δx[:] = 0
  return p
end

state(x) = x
accumulate!(x, Δ) = x

@forward Param.x Base.size

function Base.show(io::IO, p::Param)
  print(io, "Param", size(p.x))
end

# Anonymous models

export Capacitor

type Capacitor <: Model
  graph::IVertex{Any}
end

(m::Capacitor)(xs...) = interpret(reifyparams(m.graph), xs...)

graph(cap::Capacitor) = cap.graph
