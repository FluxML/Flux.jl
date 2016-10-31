module TF

using ..Flux, DataFlow, TensorFlow, Juno
import Flux: accuracy, spliceinputs, detuple

export tf

include("graph.jl")
include("model.jl")
include("recurrent.jl")

end
