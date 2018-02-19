__precompile__()

module Flux

# Zero Flux Given

using Juno, Requires
using MacroTools: @forward

export Chain, Dense, RNN, LSTM, GRU, Conv2D, Conv3D,
  Dropout, LayerNorm, BatchNorm,
  SGD, ADAM, Momentum, Nesterov, AMSGrad,
  param, params, mapleaves

using NNlib
export σ, sigmoid, logσ, logsigmoid, relu, leakyrelu, elu, swish, softmax, logsoftmax,
  conv2d, conv3d, maxpool2d, maxpool3d, avgpool2d, avgpool3d


include("tracker/Tracker.jl")
using .Tracker
export Tracker
import .Tracker: data

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

include("jit/JIT.jl")

@require CuArrays include("cuda/cuda.jl")

end # module
