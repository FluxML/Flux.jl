__precompile__()

module Flux

# Zero Flux Given

using Juno, Requires, Reexport
using MacroTools: @forward

@reexport using NNlib
using NNlib: @fix

include("tracker/Tracker.jl")
using .Tracker
using .Tracker: data
export TrackedArray, TrackedVector, TrackedMatrix, param, back!

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs
export train!,
       SGD, ADAM, Momentum, Nesterov,
       RMSProp, ADAGrad, ADADelta, AMSGrad

include("utils.jl")
include("onehot.jl")
include("treelike.jl")
export params, mapleaves, cpu, gpu, onehot, batch, glorot_normal, glorot_uniform

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")

export Chain, Dense, RNN, LSTM, GRU, Conv2D,
       Dropout, LayerNorm, BatchNorm

include("data/Data.jl")

@require CuArrays include("cuda/cuda.jl")

end # module
