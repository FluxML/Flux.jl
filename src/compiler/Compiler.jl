module Compiler

using MacroTools, DataFlow, DataFlow.Interpreter

using DataFlow: graphm, syntax, prewalk!, postwalk!, prewalk, postwalk,
  iscyclic, Constant, constant, isconstant, group, Split,
  detuple, value, inputs, thread!, value, inputs, inputnode,
  spliceinputs, bumpinputs, Line, Frame, applylines, graphinputs

include("code.jl")
include("interp.jl")
include("loops.jl")

end
