export tf, mxnet, graphviz

function loadtf()
  isdefined(Flux, :TF) && return
  @eval include(joinpath(dirname($@__FILE__), "tensorflow/tensorflow.jl"))
end

function tf(args...)
  loadtf()
  eval(:(TF.tf($(args...))))
end

function loadmx()
  isdefined(Flux, :MX) && return
  @eval include(joinpath(dirname($@__FILE__), "mxnet/mxnet.jl"))
end

function mxnet(m)
  loadmx()
  eval(:(MX.mxnet($m)))
end

include("graphviz/graphviz.jl")

const graphviz = GV.graphviz
