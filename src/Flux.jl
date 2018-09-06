module Flux

# Zero Flux Given

using MacroTools, Juno, Requires, Reexport, Statistics, Random
using MacroTools: @forward

export Chain, Dense, RNN, LSTM, GRU, Conv, MaxPool, MeanPool,
       Dropout, LayerNorm, BatchNorm,
       params, mapleaves, cpu, gpu

@reexport using NNlib

include("tracker/Tracker.jl")
using .Tracker
using .Tracker: data
export Tracker, TrackedArray, TrackedVector, TrackedMatrix, param

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs
export SGD, Descent, ADAM, AdaMax, Momentum, Nesterov,
       RMSProp, ADAGrad, ADADelta, AMSGrad

include("utils.jl")
include("onehot.jl")
include("treelike.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")

include("data/Data.jl")

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuda/cuda.jl")

end # module
