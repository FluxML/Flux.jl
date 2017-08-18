# This code is in a submodule with the intention that it will be split into an
# interface package.

module FluxCore

"""
    graph(model) => ::IVertex{Any} | nothing

Returns the graph representation of the model, if any. May be used for
compilation, generating symbolic gradients, etc.
"""
graph(m) = nothing

end
