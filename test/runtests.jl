using Flux, DataFlow, MacroTools, Base.Test
using Flux: graph, Param
using DataFlow: Input, Line

syntax(v::Vertex) = prettify(DataFlow.syntax(v))
syntax(x) = syntax(graph(x))

include("basic.jl")
include("recurrent.jl")
include("backend.jl")
