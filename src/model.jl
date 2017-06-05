"""
    back!(model, ΔY, X...) => ΔX

Backpropagate the gradient `ΔY` through the model `m`, accumulating the
gradients of any parameters. Returns the gradient of the input `X`. Gradients
may be arrays or tuples of arrays (for multiple inputs/outputs).
"""
back!(model, Δ, xs...) = error("Backprop not implemented for $(typeof(m))")

"""
    update!(model, η) => m

Update the parameters of the model `m` using the accumulated gradients from
`back!`, using the learning rate `η`.
"""
update!(m, η) = m

"""
    graph(model) => ::IVertex{Any} | nothing

Returns the graph representation of the model, if any. Most models are built
from lower-level components and can simply implement this method to get most of
Flux's functionality. If this method isn't available, functionality like
backpropagation or conversion for backend must be implemented on a case-by-case
basis. Alternatively, one can implement this method and override individual
methods as necessary.
"""
graph(m) = nothing

# Model parameters

# TODO: should be AbstractArray?
"""
A `Param` object stores a parameter array along with its gradient.
When converting to backends like TensorFlow, identical `Param`s will
result in identical variable objects.
"""
struct Param{T}
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
    update!(p::Param)

Apply the accumulated updates to the value of the parameter.
"""
function update!(p::Param, η)
  p.x .-= p.Δx .* η
  p.Δx[:] = 0
  return p
end

state(x) = x

Base.size(p::Param) = size(p.x)
Base.size(p::Param, n) = size(p.x, n)

function Base.show(io::IO, p::Param)
  print(io, "Param", size(p.x))
end

Base.copy!(xs, p::Param) = copy!(xs, p.x)
Base.copy!(p::Param, xs) = copy!(p.x, xs)

# Anonymous models

struct Capacitor
  graph::IVertex{Any}
end

(m::Capacitor)(xs...) = interpmodel(m, xs...)

graph(cap::Capacitor) = cap.graph
