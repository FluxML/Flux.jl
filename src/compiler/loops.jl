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

function collect_state(v::IVertex)
  state = typeof(v)[]
  offset = Int[]
  default = Param[]
  prewalk!(v) do v
    isa(value(v), Offset) || return v
    if (i = findfirst(state, v[1])) == 0
      push!(state, v[1])
      push!(offset, value(v).n)
      push!(default, get(value(v).default))
    else
      offset[i] = min(offset[i], value(v).n)
    end
    v
  end
  return state, offset, default
end

hiddeninput(n) = vertex(Split(n), inputnode(1))

function create_steps(v::IVertex, n)
  [bumpinputs(spliceinputs(v, hiddeninput(i))) for i = 1:n]
end

function unrollgraph(v::IVertex, n)
  state, offset, default = collect_state(v)
  v = group(group(state...), v)
  steps = create_steps(v, n)
  for i = 1:n
    vars = inputs(steps[i][1])
    prewalk!(steps[i]) do v
      isa(value(v), Offset) || return v
      stepid = value(v).n + i
      varid = findfirst(vars,v[1])
      if stepid ∈ 1:n
        steps[stepid][1,varid]
      elseif stepid < 1
        vertex(:input, constant(varid))
      elseif stepid > n
        constant(:output, constant(varid))
      end
    end
  end
  group(steps[end][1],group(map(x->x[2], steps)...))
end

unrollgraph(atomise(Chain(r,r)), 5) |> detuple |> syntax |> prettify

type Unrolled <: Model
  model
  graph::IVertex{Any}
  state::Vector{Any}
  steps::Int
end

graph(u::Unrolled) = u.graph

unroll(model, n) = Unrolled(model, unrollgraph(model, n)..., n)

@net type Recurrent
  y
  function (x)
    y = σ(x, y{-1})
  end
end

r = Recurrent(rand(5))
