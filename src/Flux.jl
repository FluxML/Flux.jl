__precompile__()

module Flux

# Zero Flux Given

using Juno, Requires, Reexport
using MacroTools: @forward

export Chain, Dense, RNN, LSTM, GRU, Conv, Conv2D,
  SGD, ADAM, Momentum, Nesterov, AMSGrad,
  param, params, mapleaves, cpu, gpu

@reexport using NNlib
using NNlib: @fix

include("tracker/Tracker.jl")
using .Tracker
export Tracker
import .Tracker: data

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs

include("utils.jl")
include("onehot.jl")
include("treelike.jl")

include("layers/loss.jl")
export cross_entropy, bce, mse, bce_logit, nll

include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")

include("layers/normalisation.jl")
export Dropout, LayerNorm, BatchNorm, normalise

include("data/Data.jl")

include("jit/JIT.jl")

@require CuArrays include("cuda/cuda.jl")

end # module
