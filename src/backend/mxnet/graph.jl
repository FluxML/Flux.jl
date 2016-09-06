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
    !isa(x, Flux.ModelInput) ? x :
    isa(x.name, Integer) ? args[x.name] : getfield(model, x.name)
  end
  postwalk(g) do v
    vertex(graph(vars, cvalue(v), cvalue.(inputs(v))...))
  end |> value
end

type SoftmaxOutput
  name::Symbol
end

function rewrite_softmax(model, name)
  model == softmax && return SoftmaxOutput(name)
  g = Flux.graph(model)
  (g == nothing || value(g) ≠ softmax || Flow.nin(g) ≠ 1) && error("mx.FeedForward models must end with `softmax`")
  return Flux.Capacitor(vertex(SoftmaxOutput(name), g[1]))
end

# Built-in implemenations

node(::typeof(*), args...) = mx.dot(args...)
node(::typeof(+), args...) = mx.broadcast_plus(args...)
node(::typeof(σ), x) = mx.Activation(data = x, act_type = :sigmoid)
node(::typeof(relu), x) = mx.Activation(data = x, act_type=:relu)
node(::typeof(tanh), x) = mx.Activation(data = x, act_type=:tanh)
node(::typeof(flatten), x) = mx.Flatten(data = x)

node(::typeof(softmax), xs) =
  mx.broadcast_div(exp(xs), mx.Reshape(mx.sum(exp(xs)), shape = (1,1)))

node(s::SoftmaxOutput, xs) = mx.SoftmaxOutput(data = xs, name = s.name)

graph(vars, ::Input, x) = x

graph(vars, c::Conv, x) =
  mx.Convolution(data = x,
                 kernel = c.size,
                 num_filter = c.features,
                 stride = c.stride)

graph(vars, p::MaxPool, x) =
  mx.Pooling(data = x,
             pool_type = :max,
             kernel = p.size,
             stride = p.stride)

# TODO: fix the initialisation issue
graph(vars, d::Dense, x) =
  mx.FullyConnected(data = x,
                    num_hidden = size(d.W.x, 1),
                    # weight = graph(vars, d.W),
                    # bias = graph(vars, d.b)
                    )
