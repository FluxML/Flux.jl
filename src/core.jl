# This code is in a submodule with the intention that it will be split into an
# interface package.

module FluxCore

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

end
