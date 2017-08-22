__precompile__()

module Flux

using Juno
using Lazy: @forward

export Chain, Linear

# Zero Flux Given

using NNlib
export σ, relu, softmax

include("Tracker/Tracker.jl")
using .Tracker
export track, back!

include("utils.jl")

include("compiler/Compiler.jl")
using .Compiler: @net

include("layers/stateless.jl")
include("layers/basic.jl")

end # module
