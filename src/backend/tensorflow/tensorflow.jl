module TF

using ..Flux, DataFlow, TensorFlow, Juno
import Flux: accuracy, spliceinputs, detuple

export tf

type Op
  f
  shape
end

Op(f) = Op(f, (d...) -> nothing)

include("graph.jl")
include("model.jl")
include("recurrent.jl")

end
