module TF

using ..Flux, Flow, TensorFlow, Juno
import Flux: accuracy, spliceinputs, detuple

export tf

include("graph.jl")
include("model.jl")

end
