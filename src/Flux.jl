__precompile__()

module Flux

using MacroTools, Lazy, DataFlow, Juno
using DataFlow: graphm, syntax, prewalk!, postwalk!, prewalk, postwalk,
  iscyclic, Constant, constant, isconstant, group, Split, splitnode,
  detuple, value, inputs, thread!, value, inputs, Split, splitnode, inputnode,
  spliceinputs, bumpinputs, Line, Frame, applylines, graphinputs
using DataFlow.Interpreter

export @net, unroll, unroll1, @shapes,
  @Chain, Chain, Input, Affine, Conv2D, Recurrent, GatedRecurrent, LSTM,
  σ, relu, softmax,
  tf, mxnet

# Zero Flux Given

include("Tracker/Tracker.jl")
using .Tracker

include("utils.jl")
include("params.jl")

include("compiler/code.jl")
include("compiler/loops.jl")
include("compiler/interp.jl")

include("layers/control.jl")
include("layers/affine.jl")
include("layers/activation.jl")
include("layers/cost.jl")
include("layers/recurrent.jl")

include("data.jl")

end # module
