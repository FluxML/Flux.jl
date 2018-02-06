__precompile__()

module Flux

# Zero Flux Given

using Juno, Requires
using MacroTools: @forward

export Chain, Dense, RNN, LSTM, GRU, Conv2D,
  Dropout, LayerNorm, BatchNorm,
  SGD, ADAM, Momentum, Nesterov, AMSGrad,
  param, params, mapleaves

using NNlib
export σ, sigmoid, logσ, logsigmoid, relu, leakyrelu, elu, swish, softmax, logsoftmax,
  conv2d, maxpool2d, avgpool2d

include("tracker/Tracker.jl")
using .Tracker
export Tracker
import .Tracker: data, value

include("optimise/Optimise.jl")
using .Optimise

include("utils.jl")
include("onehot.jl")
include("treelike.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalisation.jl")

include("data/Data.jl")

end # module
