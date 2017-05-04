using Flux, DataFlow, MacroTools, Base.Test
using Flux: graph, Param, unsqueeze
using DataFlow: Line, Frame

macro mxonly(ex)
  :(Base.find_in_path("MXNet") ≠ nothing && $(esc(ex)))
end

macro tfonly(ex)
  :(Base.find_in_path("TensorFlow") ≠ nothing && $(esc(ex)))
end

include("batching.jl")
include("backend/common.jl")

include("basic.jl")
include("recurrent.jl")
@tfonly include("backend/tensorflow.jl")
@mxonly include("backend/mxnet.jl")
