export unroll, unroll1

struct Offset
  name::Symbol
  n::Int
  default::Nullable{Any}
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

create_steps(v::IVertex, n) = [bumpinputs(spliceinputs(v, hiddeninput(i))) for i = 1:n]

function getvar(n, step, steps, offset, default)
  if step < 1
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

# Input:  (hidden1, hidden2, ...), (x1, x2, ...)
# Output: (hidden1, hidden2, ...), (y1, y2, ...)

function unrollgraph(v::IVertex, n)
  state, offset, default = collect_state(v)
  v = group(group(state...), v)
  steps = create_steps(v, n)
  for i = 1:n
    vars = inputs(steps[i][1])
    postwalk!(steps[i]) do v
      value(v) isa Offset || return v
      varid = findfirst(vars,v[1])
      getvar(varid, value(v).n + i, steps, offset, default)
    end
  end
  out = group(map(x->x[2], steps)...)
  state, defaults = stateout(steps, offset, default)
  group(state,out), map(Flux.state, defaults)
end

unrollgraph(m, n; kws...) = unrollgraph(atomise(m), n; kws...)

function unroll(model, n)
  graph, state = unrollgraph(model, n)
  SeqModel(Stateful(Capacitor(graph), state), n)
end

function unseqin(v::IVertex)
  prewalk(v) do v
    # TODO: inputidx function
    isa(value(v), Split) && DataFlow.isinput(v[1]) && value(v[1]).n == 2 ? v[1] : v
  end
end

unseqout(v::IVertex) = group(v[1], map(x->x[1], inputs(v)[2:end])...)

unseq(graph) = unseqout(unseqin(graph))

function unroll1(model)
  graph, state = unrollgraph(model, 1)
  Stateful(Capacitor(unseq(graph)), state)
end

flip(model) = Capacitor(map(x -> x isa Offset ? -x : x, atomise(model)))
