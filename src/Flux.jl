__precompile__()

module Flux

# Zero Flux Given

using Juno, Requires
using Lazy: @forward

export AbstractLayer, Chain, Dense, RNN, LSTM,
  SGD, params, mapparams

using NNlib
export σ, relu, softmax

include("tracker/Tracker.jl")
using .Tracker

include("optimise/Optimise.jl")
using .Optimise

include("utils.jl")
include("onehot.jl")
include("tree.jl")

include("layers/AbstractLayer.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/recurrent.jl")

end # module
