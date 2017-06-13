# This code is in a submodule with the intention that it will be split into an
# interface package.

module FluxCore

"""
    back!(model, ΔY, X...) => ΔX

Backpropagate the gradient `ΔY` through the model `model`, accumulating the
gradients of any parameters. Returns the gradient of the input `X`. Gradients
may be arrays or tuples of arrays (for multiple inputs/outputs).
"""
back!(model, Δ, xs...) = error("Backprop not implemented for $(typeof(model))")

"""
    update!(model, o) => model

Update the parameters of the model `model` using the accumulated gradients from
`back!`, using the optimizer `o`.
"""
update!(m, o) = m

"""
    graph(model) => ::IVertex{Any} | nothing

Returns the graph representation of the model, if any. May be used for
compilation, generating symbolic gradients, etc.
"""
graph(m) = nothing

end
