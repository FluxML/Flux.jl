export tf

function loadtf()
  isdefined(Flux, :TF) && return
  @eval include(joinpath(dirname($@__FILE__), "tensorflow/tensorflow.jl"))
end

function tf(args...)
  loadtf()
  TF.tf(args...)
end
