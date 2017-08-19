__precompile__()

module Flux

using Juno
using Lazy: @forward

export Chain, Affine, σ, softmax

# Zero Flux Given

include("Tracker/Tracker.jl")
using .Tracker
export track, back!

include("utils.jl")
include("params.jl")

include("compiler/Compiler.jl")
using .Compiler: @net

include("layers/chain.jl")
include("layers/affine.jl")
include("layers/activation.jl")
include("layers/cost.jl")

include("data.jl")

end # module
