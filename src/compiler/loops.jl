export unroll, unroll1

type Offset
  name::Symbol
  n::Int
  default::Nullable{Param}
end

Offset(name, n) = Offset(name, n, nothing)

Base.:-(o::Offset) = Offset(o.name, -o.n, o.default)

function liftloops(ex)
  ex = DataFlow.normedges(ex)
  decls = Dict()
  ex = MacroTools.postwalk(ex) do ex
    @capture(ex, x_{n_}) || return ex
    haskey(decls, (x,n)) && return namify(decls[(x,n)])
    @gensym edge
    decls[(x,n)] = :($edge = $(Offset(x,n))($x))
    edge
  end
  prepend!(ex.args, collect(values(decls)))
  ex
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
    value(v) isa Offset || return v
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

function create_steps(v::IVertex, n; seq = true, stateful = true)
  [(stateful ? bumpinputs : copy)(seq ? spliceinputs(v, hiddeninput(i)) : v) for i = 1:n]
end

function getvar(n, step, steps, offset, default; stateful = true)
  if stateful && step < 1
    hiddeninput(sum(offset[1:n-1]) + 1 - step)
  elseif step ∉ 1:length(steps)
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

function unrollgraph(v::IVertex, n; seq = true, stateful = true)
  state, offset, default = collect_state(v)
  v = group(group(state...), v)
  steps = create_steps(v, n, seq = seq, stateful = stateful)
  for i = 1:n
    vars = inputs(steps[i][1])
    postwalk!(steps[i]) do v
      value(v) isa Offset || return v
      varid = findfirst(vars,v[1])
      getvar(varid, value(v).n + i, steps, offset, default, stateful = stateful)
    end
  end
  out = group(map(x->x[2], steps)...)
  if stateful
    state, defaults = stateout(steps, offset, default)
    group(state,out), map(Flux.state, defaults)
  else
    out, []
  end
end

unrollgraph(m, n; kws...) = unrollgraph(atomise(m), n; kws...)

# TODO: perhaps split into SeqModel + StatefulModel
type Unrolled <: Model
  model
  graph::IVertex{Any}
  state::Vector{Any}
  stateful::Bool
  steps::Int
end

(m::Unrolled)(xs...) = interpret(reifyparams(m.graph), xs...)

graph(u::Unrolled) = u.graph

function unroll(model, n; seq = true, stateful = true)
  graph, state = unrollgraph(model, n; seq = seq, stateful = stateful)
  seq || stateful ? Unrolled(model, graph, state, stateful, n) : Capacitor(graph)
end

function unroll1(model)
  graph, state = unrollgraph(model, 1; seq = false)
  graph = group(graph[1], map(x->x[1], inputs(graph)[2:end])...)
  Unrolled(model, graph, state, false, 1)
end

flip(model) = Capacitor(map(x -> x isa Offset ? -x : x, atomise(model)))
