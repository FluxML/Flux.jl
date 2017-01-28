module MX

using MXNet, DataFlow, ..Flux

export mxnet

include("graph.jl")
include("model.jl")

end
