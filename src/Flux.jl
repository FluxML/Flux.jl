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
export SGD, ADAM, AdaMax, Momentum, Nesterov,
       RMSProp, ADAGrad, ADADelta, AMSGrad, NADAM

include("utils.jl")
include("onehot.jl")
include("treelike.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/regularisedchain.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalise.jl")

include("data/Data.jl")

@require CuArrays include("cuda/cuda.jl")

end # module
