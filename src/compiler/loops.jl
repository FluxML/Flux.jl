export unroll

type Offset
  name::Symbol
  n::Int
  default::Nullable{Param}
end

Offset(name, n) = Offset(name, n, nothing)

function liftloops(ex, params)
  ex = DataFlow.normedges(ex)
  MacroTools.postwalk(ex) do ex
    @capture(ex, x_{n_}) || return ex
    :($(Offset(x,n))($x))
  end
end

function hasloops(model)
  g = graph(model)
  g == nothing && return false
  iscyclic(g) && return true
  result = false
  map(m -> hasloops(m) && (result = true), g)
  return result
end

function atomise(model)
  postwalk(graph(model)) do v
    hasloops(value(v)) || return v
    spliceinputs(atomise(value(v)), inputs(v)...)
  end
end

# hiddeninput(n) = vertex(Split(n), inputnode(1))
#
# function unroll!(delay::IVertex, n)
#   prewalk!(delay[1]) do v
#     v === delay ? hiddeninput(n) : v
#   end
# end
#
# function break!(g::IVertex)
#   g = bumpinputs(g)
#   loops = []
#   defaults = []
#   g = prewalk!(g) do v
#     isa(value(v), Offset) || return v
#     n = length(loops)+1
#     push!(loops, unroll!(v, n))
#     push!(defaults, get(value(v).default))
#     hiddeninput(n)
#   end
#   cse(group(group(loops...), g)), defaults
# end
#
# function unrollgraph(model, n)
#   graph, defaults = break!(atomise(model))
#   outputs = [spliceinputs(graph, group([constant(splitnode(inputnode(1),i)) for i = 1:length(defaults)]...),
#                                  splitnode(inputnode(2), 1))]
#   for i = 2:n
#     push!(outputs, spliceinputs(graph, outputs[end][1], splitnode(inputnode(2), i)))
#   end
#   state = outputs[end][1]
#   outputs = map(x -> x[2], outputs)
#   (@> group(state, group(outputs...)) detuple), map(x->x.x, defaults)
# end

type Unrolled <: Model
  model
  graph::IVertex{Any}
  state::Vector{Any}
  steps::Int
end

graph(u::Unrolled) = u.graph

unroll(model, n) = Unrolled(model, unrollgraph(model, n)..., n)
