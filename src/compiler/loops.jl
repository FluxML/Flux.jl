export unroll

type Offset
  name::Symbol
  n::Int
  default::Nullable{Param}
end

Offset(name, n) = Offset(name, n, nothing)

Base.:-(o::Offset) = Offset(o.name, -o.n, o.default)

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
      push!(offset, max(0, -value(v).n))
      push!(default, get(value(v).default))
    else
      offset[i] = max(offset[i], -value(v).n)
    end
    v
  end
  return state, offset, default
end

hiddeninput(n) = vertex(Split(n), inputnode(1))

function create_steps(v::IVertex, n)
  [bumpinputs(spliceinputs(v, hiddeninput(i))) for i = 1:n]
end

function getvar(n, step, steps, offset, default)
  if step < 1
    hiddeninput(sum(offset[1:n-1]) + 1 - step)
  elseif step > length(steps)
    constant(default[n])
  else
    steps[step][1,n]
  end
end

function stateout(steps, offset, default)
  outs = []
  defaults = []
  for i = 1:length(offset), j = 1:offset[i]
    push!(outs, getvar(i, length(steps)-j+1, steps, offset, default))
    push!(defaults, default[i])
  end
  group(outs...), defaults
end

function unrollgraph(v::IVertex, n)
  state, offset, default = collect_state(v)
  v = group(group(state...), v)
  steps = create_steps(v, n)
  for i = 1:n
    vars = inputs(steps[i][1])
    postwalk!(steps[i]) do v
      isa(value(v), Offset) || return v
      varid = findfirst(vars,v[1])
      getvar(varid, value(v).n + i, steps, offset, default)
    end
  end
  state, defaults = stateout(steps, offset, default)
  group(state,group(map(x->x[2], steps)...)), map(Flux.state, defaults)
end

unrollgraph(m, n) = unrollgraph(atomise(m), n)

type Unrolled <: Model
  model
  graph::IVertex{Any}
  state::Vector{Any}
  steps::Int
end

graph(u::Unrolled) = u.graph

unroll(model, n) = Unrolled(model, unrollgraph(model, n)..., n)

flip(model) = Capacitor(map(x -> isa(x, Offset) ? -x : x, atomise(model)))
