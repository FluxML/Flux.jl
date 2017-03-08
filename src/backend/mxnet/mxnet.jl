module MX

using MXNet, DataFlow, ..Flux

export mxnet

include("mxarray.jl")
include("graph.jl")
include("model.jl")

end
