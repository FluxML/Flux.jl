# We use a lazy-loading trick to load the backend code as needed; this avoids
# the need for a hard dependency on both backends.

# This is effectively equivalent to:
#   include("tensorflow/tensorflow.jl")
#   using .TF
#   export tf
# but instead of loading immediately, we wait until `tf` is first called.

function loadtf()
  isdefined(Flux, :TF) && return
  @eval include(joinpath(dirname($@__FILE__), "tensorflow/tensorflow.jl"))
end

function tf(args...)
  loadtf()
  eval(:(TF.tf($(QuoteNode.(args)...))))
end

function loadmx()
  isdefined(Flux, :MX) && return
  @eval include(joinpath(dirname($@__FILE__), "mxnet/mxnet.jl"))
end

function mxnet(args...)
  loadmx()
  eval(:(MX.mxnet($(QuoteNode.(args)...))))
end
