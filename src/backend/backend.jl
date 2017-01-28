export tf, mxnet

function loadtf()
  isdefined(Flux, :TF) && return
  @eval include(joinpath(dirname($@__FILE__), "tensorflow/tensorflow.jl"))
end

function tf(args...)
  loadtf()
  TF.tf(args...)
end

function loadmx()
  isdefined(Flux, :MX) && return
  @eval include(joinpath(dirname($@__FILE__), "mxnet/mxnet.jl"))
end

function mxnet(args...)
  loadmx()
  MX.mxnet(args...)
end
