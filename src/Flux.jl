module Flux

using MacroTools, Lazy, DataFlow, Juno
using DataFlow: graphm, syntax, prewalk!, postwalk!, prewalk, postwalk,
  iscyclic, Constant, constant, isconstant, group, Split, splitnode,
  detuple, value, inputs, thread!, value, inputs, Split, splitnode, inputnode,
  spliceinputs, bumpinputs, Frame
using Juno: Tree, Row

# Zero Flux Given

include("dims/catmat.jl")
include("dims/batching.jl")
include("dims/seq.jl")

include("model.jl")
include("utils.jl")
include("data.jl")

include("compiler/code.jl")
include("compiler/loops.jl")
include("compiler/interp.jl")
include("compiler/shape.jl")

include("layers/affine.jl")
include("layers/activation.jl")
include("layers/recurrent.jl")
include("layers/shape.jl")
include("layers/chain.jl")
include("layers/shims.jl")

include("cost.jl")

include("backend/backend.jl")

end # module
