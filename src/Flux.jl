__precompile__()

module Flux

# Zero Flux Given

using Juno, Requires, Reexport
using MacroTools: @forward

export Chain, Dense, RNN, LSTM, GRU, Conv,
       Dropout, LayerNorm, BatchNorm,
       params, mapleaves, cpu, gpu

@reexport using NNlib
using NNlib: @fix

include("tracker/Tracker.jl")
using .Tracker
using .Tracker: data
export Tracker, TrackedArray, TrackedVector, TrackedMatrix, param

include("optimise/Optimise.jl")
using .Optimise
using .Optimise: @epochs
export SGD, ADAM, ADAMW, AdaMax, Momentum, Nesterov,
       RMSProp, ADAGrad, ADADelta, AMSGrad, NADAM

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
