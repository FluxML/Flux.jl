cvalue(x) = x
cvalue(c::Constant) = c.value
cvalue(v::Vertex) = cvalue(value(v))

graph(vars, model, args...) = node(model, args...)

graph(vars, x::mx.SymbolicNode) = x

# TODO: detect parameters used more than once
function graph{T<:AArray}(vars, p::Flux.Param{T})
  value = p.x
  id = gensym()
  vars[id] = value
  return mx.Variable(id)
end

function graph(vars, model::Model, args...)
  g = Flux.graph(model)
  g = Flow.mapconst(g) do x
    !isa(x, Flux.Parameter) ? x :
    isa(x.name, Integer) ? args[x.name] : getfield(model, x.name)
  end
  postwalk(g) do v
    vertex(graph(vars, cvalue(v), cvalue.(inputs(v))...))
  end |> value
end

# Built-in implemenations

node(::typeof(*), args...) = mx.dot(args...)
node(::typeof(+), args...) = mx.broadcast_plus(args...)
node(::typeof(σ), x) = mx.Activation(data = x, act_type = :sigmoid)
