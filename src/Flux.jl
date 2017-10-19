__precompile__()

module Flux

# Zero Flux Given

using Juno, Requires
using Lazy: @forward

export Chain, Dense, RNN, LSTM,
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
include("tree.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/recurrent.jl")

end # module
