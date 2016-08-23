module Flux

using MacroTools, Lazy, Flow

# Zero Flux Given

include("model.jl")
include("utils.jl")

include("compiler/diff.jl")
include("compiler/code.jl")

include("cost.jl")
include("activation.jl")
include("layers/input.jl")
include("layers/dense.jl")
include("layers/sequence.jl")

include("backend/mxnet/mxnet.jl")

end # module
