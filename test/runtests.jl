using Flux, DataFlow, MacroTools, Base.Test
using Flux: graph, Param
using DataFlow: Line, Frame

syntax(v::Vertex) = prettify(DataFlow.syntax(v))
syntax(x) = syntax(graph(x))

macro mxonly(ex)
  :(Base.find_in_path("MXNet") ≠ nothing && $(esc(ex)))
end

macro tfonly(ex)
  :(Base.find_in_path("TensorFlow") ≠ nothing && $(esc(ex)))
end

include("batching.jl")
include("basic.jl")
include("recurrent.jl")
include("backend.jl")
