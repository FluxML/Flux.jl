__precompile__()

module Flux

# Zero Flux Given

using Juno, Requires
using Lazy: @forward

export Chain, Dense, RNN, LSTM, Dropout,
  SGD, ADAM, Momentum, Nesterov,
  param, params, mapleaves

using NNlib
export σ, relu, leakyrelu, elu, swish, softmax

include("tracker/Tracker.jl")
using .Tracker

include("optimise/Optimise.jl")
using .Optimise

include("utils.jl")
include("onehot.jl")
include("treelike.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/recurrent.jl")
include("layers/normalisation.jl")

include("batches/Batches.jl")
include("data/Data.jl")

end # module
