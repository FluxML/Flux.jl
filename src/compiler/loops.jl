type Delay
  name::Symbol
  default::Nullable{Param}
end

Delay(name) = Delay(name, nothing)

function liftloops!(ex, params)
  e = Flow.normedges(ex)
  hidden = intersect((b.args[1] for b in ex.args), params)
  edges = Dict(h => gensym("edge") for h in hidden)
  for b in ex.args
    b.args[2] = MacroTools.postwalk(x -> get(edges, x, x), b.args[2])
  end
  for (h, e) in edges
    unshift!(ex.args, :($e = $(Delay(h))($h)))
  end
  return ex
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

hiddeninput(n) = vertex(Split(n), inputnode(1))

function unroll!(delay::IVertex, n)
  prewalk!(delay[1]) do v
    v === delay ? hiddeninput(n) : v
  end
end

function break!(g::IVertex)
  g = bumpinputs(g)
  loops = []
  defaults = []
  g = prewalk!(g) do v
    isa(value(v), Delay) || return v
    n = length(loops)+1
    push!(loops, unroll!(v, n))
    push!(defaults, get(value(v).default))
    hiddeninput(n)
  end
  cse(vertex(tuple, vertex(tuple, loops...), g)), defaults
end

# function unroll(model, n)
#   graph, defaults = break!(atomise(model))
#   outputs = [spliceinputs(graph, vertex(tuple, map(constant, defaults)...), inputnode(1))]
#   for i = 2:n
#     push!(outputs, spliceinputs(graph, outputs[end][1], constant(ModelInput(i))))
#   end
#   state = outputs[end][1]
#   outputs = map(x -> x[2], outputs)
#   vertex(tuple, state, vertex(tuple, outputs...))
# end

# r = Recurrent(10,10)
# unroll(r, 1) |> syntax |> prettify |> display

@net type Recurrent
  Wx; Wh; B
  hidden

  function (x)
    hidden = σ( Wx*x + Wh*hidden + B )
  end
end

Recurrent(in::Integer, out::Integer; init = initn) =
  Recurrent(init(out, in), init(out, out), init(out), zeros(out))

Base.show(io::IO, r::Recurrent) =
  print(io, "Flux.Recurrent(...)")
