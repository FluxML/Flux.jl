# Stateful Models

mutable struct Stateful
  model
  states::Vector{Any}
  istate::Vector{Any}
  ostate::Vector{Any}
end

Stateful(model, ss) = Stateful(model, ss, state.(ss), state.(ss))

function (m::Stateful)(xs...)
  m.istate = m.ostate
  state, y = m.model((m.istate...,), xs...)
  m.ostate = collect(state)
  return y
end

function back!(m::Stateful, Δ, x)
  back!(m.model, ((zeros.(m.ostate)...,), Δ), (m.istate...,), x)[2:end]
end

update!(m::Stateful, η) = update!(m.model, η)

# Seq Models

struct SeqModel
  model
  steps::Int
end

seqtuple(x, n) = x
seqtuple(xs::Tuple, n) = seqtuple.(xs, n)

seqtuple(xs::AbstractArray, n) =
  ndims(xs) < 3 ? xs :
  n ≠ 0 && size(xs, 2) ≠ n ? error("Expecting sequence length $n, got $(size(xs, 2))") :
  (unstack(xs, 2)...)

seqtuple(xs::Batch{<:Seq}, n) = seqtuple(rawbatch(xs), n)

reseq(x) = x
reseq(x::Tuple{}) = ()
reseq(xs::Tuple) = all(isa.(xs, AbstractArray) .& (ndims.(xs) .≥ 2)) ? stack(xs, 2) : reseq.(xs)

function (m::SeqModel)(xs...)
  xs = seqtuple(xs, m.steps)
  reseq(m.model(xs...))
end

function back!(m::SeqModel, args...)
  args = seqtuple(args, 0)
  # TODO: reseq
  back!(m.model, args...)
end

update!(m::SeqModel, η) = update!(m.model, η)

graph(m::SeqModel) = graph(m.model)

# Recurrent Graphs

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
  default = []
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

hiddeninput(n, t) = vertex(Split(t), inputnode(n))

# TODO: nicer way to do this.
create_steps(v::IVertex, n) = [bumpinputs(spliceinputs(v, [hiddeninput(n, t) for n = 1:graphinputs(v)]...)) for t = 1:n]

function getvar(n, step, steps, offset, default)
  if step < 1
    hiddeninput(1, sum(offset[1:n-1]) + 1 - step)
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
# TODO: make sure there's a reasonable order for hidden states

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
  group(state,out), defaults
end

unrollgraph(m, n; kws...) = unrollgraph(atomise(m), n; kws...)

function unroll(model, n)
  graph, state = unrollgraph(model, n)
  SeqModel(Stateful(Capacitor(graph), state), n)
end

function stateless(s::Stateful)
  v = graph(s.model)
  v = spliceinputs(v, group(constant.(s.states)...),
                   [inputnode(i) for i = 1:graphinputs(v)-1]...)
  Capacitor(v[2])
end

stateless(s::SeqModel) = SeqModel(stateless(s.model), s.steps)

function unseqin(v::IVertex)
  prewalk(v) do v
    # TODO: inputidx function
    isa(value(v), Split) && DataFlow.isinput(v[1]) && value(v[1]).n > 1 ? v[1] : v
  end
end

unseqout(v::IVertex) = group(v[1], v[2][1])

unseq(graph) = unseqout(unseqin(graph))

function unroll1(model)
  graph, state = unrollgraph(model, 1)
  Stateful(Capacitor(unseq(graph)), state)
end

flip(model) = Capacitor(map(x -> x isa Offset ? -x : x, atomise(model)))
