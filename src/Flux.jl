__precompile__()

module Flux

# Zero Flux Given

using Juno
using Lazy: @forward

export Chain, Dense, RNN, LSTM,
  SGD

using NNlib
export σ, relu, softmax

include("tracker/Tracker.jl")
using .Tracker

include("optimise/Optimise.jl")
using .Optimise

include("utils.jl")

include("compiler/Compiler.jl")
using .Compiler: @net

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/recurrent.jl")

end # module
