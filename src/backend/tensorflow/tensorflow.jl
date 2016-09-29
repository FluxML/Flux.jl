module TF

using ..Flux, Flow, TensorFlow
import Juno: info
import Flux: accuracy

export tf

# Workaround for tensor display bug
using Juno
Media.render(::Juno.Clipboard, ::Tensor) = "Tensor()"

cvalue(x) = x
cvalue(c::Constant) = c.value
cvalue(v::Vertex) = cvalue(value(v))

graph(x::Tensor) = x

# TODO: detect variable reuse
graph{T<:AArray}(p::Flux.Param{T}) = Variable(p.x')

function graph(model::Model, args...)
  g = Flux.graph(model)
  g = Flow.mapconst(g) do x
    !isa(x, Flux.ModelInput) ? x :
    isa(x.name, Integer) ? args[x.name] : getfield(model, x.name)
  end
  postwalk(g) do v
    vertex(graph(cvalue(v), cvalue.(inputs(v))...))
  end |> value
end

graph(::typeof(*), args...) = *(reverse(args)...)
graph(::typeof(+), args...) = +(args...)
graph(::typeof(softmax), x) = nn.softmax(x)

type Model
  session::Session
  inputs::Vector{Tensor}
  graph::Tensor
  grad::Tensor
end

Media.render(::Juno.Clipboard, ::Model) = "Flux.TF.Model()"

function tf(model)
  sess = Session(Graph())
  input = placeholder(Float64)
  g = graph(model, input)
  run(sess, initialize_all_variables())
  Model(sess, [input], g, gradients(g, input))
end

function (m::Model)(args...)
  @assert length(args) == length(m.inputs)
  run(m.session, m.graph, Dict(zip(m.inputs, map(transpose, args))))'
end

function Flux.back!(m::Model, Δ, args...)
  @assert length(args) == length(m.inputs)
  # TODO: keyword arguments to `gradients`
  run(m.session, m.grad, Dict(zip(m.inputs, args)))
end

function Flux.update!(m::Model)
  error("update! is not yet supported on TensorFlow models")
end

function Flux.train!(m::Model, train, test=[]; epoch = 1, η = 0.1,
                     loss = (y, y′) -> reduce_sum((y - y′).^2)/2,
                     opt = TensorFlow.train.GradientDescentOptimizer(η))
  i = 0
  Y = placeholder(Float64)
  Loss = loss(m.graph, Y)
  minimize_op = TensorFlow.train.minimize(opt, Loss)
  run(m.session, initialize_all_variables())
  for e in 1:epoch
    info("Epoch $e\n")
    @progress for (x, y) in train
      y, cur_loss, _ = run(m.session, vcat(m.graph, Loss, minimize_op), Dict(m.inputs[1]=>x', Y=>y'))
      i % 1000 == 0 && @show accuracy(m, test)
      i += 1
    end
  end
end

end
