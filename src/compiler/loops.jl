type Delay
  name::Symbol
end

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

hinput(n) = vertex(getindex, constant(ModelInput(1)), constant(n))

function unroll!(delay::IVertex, n)
  prewalk!(delay[1]) do v
    v === delay ? hinput(n) : v
  end
end

function break!(g::IVertex)
  g = bumpinputs(g)
  loops = []
  g = prewalk!(g) do v
    isa(value(v), Delay) || return v
    n = length(loops)+1
    push!(loops, unroll!(v, n))
    hinput(n)
  end
  cse(vertex(tuple, vertex(tuple, loops...), g))
end

# r = Recurrent(10, 10)
# r = Chain(Dense(10,10), Recurrent(10,10))
# r = Chain(Recurrent(10,10),Recurrent(10,10))

# break!(atomise(r)) |> syntax |> prettify |> display

# @model type Recurrent
#   Wx; Wh; B
#   hidden
#
#   function (x)
#     hidden = σ( Wx*x + Wh*hidden + B )
#   end
# end
#
# Recurrent(in::Integer, out::Integer; init = initn) =
#   Recurrent(init(out, in), init(out, out), init(out), zeros(out))

@model type Recurrent
  model
  hidden
  function (x)
    hidden = σ(model(vcat(x, hidden)))
  end
end

Recurrent(in::Integer, out::Integer; init = initn) =
  Recurrent(Dense(in + out, out, init = init), zeros(out))

Base.show(io::IO, r::Recurrent) =
  print(io, "Flux.Recurrent(...)")
