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

@net type TLP
  first
  second
  function (x)
    l1 = σ(first(x))
    l2 = softmax(second(l1))
  end
end

@net type Multi
  W
  V
  (x, y) -> (x*W, y*V)
end

Multi(in::Integer, out::Integer) =
  Multi(randn(in, out), randn(in, out))

include("batching.jl")
include("basic.jl")
include("recurrent.jl")
@tfonly include("backend/tensorflow.jl")
@mxonly include("backend/mxnet.jl")
