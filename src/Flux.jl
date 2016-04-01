module Flux

using MacroTools, Lazy, Flow

# Zero Flux Given

export Model, back!, update!

abstract Model
abstract Activation <: Model

back!(m::Model, ∇) = error("Backprop not implemented for $(typeof(m))")
update!(m::Model, η) = m

include("capacitor.jl")

include("compiler/diff.jl")
include("compiler/loop.jl")
include("compiler/code.jl")

include("cost.jl")
include("activation.jl")
include("layers/input.jl")
include("layers/dense.jl")
include("layers/sequence.jl")
include("utils.jl")

end # module
