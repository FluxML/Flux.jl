__precompile__()

module Flux

using MacroTools, Lazy, DataFlow, Juno
using DataFlow: graphm, syntax, prewalk!, postwalk!, prewalk, postwalk,
  iscyclic, Constant, constant, isconstant, group, Split, splitnode,
  detuple, value, inputs, thread!, value, inputs, Split, splitnode, inputnode,
  spliceinputs, bumpinputs, Line, Frame, applylines, graphinputs
using DataFlow.Interpreter
using Juno: Tree, Row

# Zero Flux Given

include("utils.jl")

include("model.jl")

include("dims/catmat.jl")
include("dims/batching.jl")
include("dims/seq.jl")
include("dims/iter.jl")

include("compiler/code.jl")
include("compiler/loops.jl")
include("compiler/interp.jl")
include("compiler/shape.jl")

include("layers/control.jl")
include("layers/affine.jl")
include("layers/activation.jl")
include("layers/cost.jl")
include("layers/recurrent.jl")
include("layers/shims.jl")

include("backend/backend.jl")

include("data.jl")
include("training.jl")

end # module
