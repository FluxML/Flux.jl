__precompile__()

module Flux

using Juno
using Lazy: @forward

export Chain, Linear, σ, softmax

# Zero Flux Given

include("Tracker/Tracker.jl")
using .Tracker
export track, back!

include("utils.jl")
include("params.jl")

include("compiler/Compiler.jl")
using .Compiler: @net

include("layers/stateless.jl")
include("layers/basic.jl")

end # module
