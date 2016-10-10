module TF

using ..Flux, Flow, TensorFlow, Juno
import Flux: accuracy
import Juno: info

export tf

cvalue(x) = x
cvalue(c::Constant) = c.value
cvalue(v::Vertex) = cvalue(value(v))

graph(x::Tensor) = x

# TODO: detect variable reuse
graph{T<:AArray}(p::Flux.Param{T}) = Variable(p.x)

function graph(model::Model, args...)
  g = Flux.graph(model)
  g ≠ nothing || error("No graph for $model")
  g = Flow.mapconst(g) do x
    !isa(x, Flux.ModelInput) ? x :
    isa(x.name, Integer) ? args[x.name] : getfield(model, x.name)
  end
  postwalk(g) do v
    vertex(graph(cvalue(v), cvalue.(inputs(v))...))
  end |> value
end

graph(::typeof(*), args...) = *(args...)
graph(::typeof(+), args...) = +(args...)
graph(::typeof(softmax), x) = nn.softmax(x)
graph(::typeof(relu), x) = nn.relu(x)
graph(::typeof(tanh), x) = tanh(x)

# reshape hack due to https://github.com/malmaud/TensorFlow.jl/issues/79
batchsize(x::Tensor) = reduce_sum(slice(TensorFlow.shape(x), [0], [1]))
graph(::typeof(flatten), x) = reshape(x, pack([batchsize(x), Int32(-1)]))
graph(r::Reshape, x) = reshape(x, pack([batchsize(x), map(Int32, r.dims)...]))

graph(::Input, x) = x

graph(c::Conv2D, x) =
  nn.conv2d(x, graph(c.filter), [1,c.stride...,1], "VALID")

graph(p::MaxPool, x) =
  nn.max_pool(x, [1, p.size..., 1], [1, p.stride..., 1], "VALID")

TensorFlow.Tensor(m::Flux.Model, args...) = graph(m, args...)

# Treat the first dimension as the batch index
# TODO: custom data type for this
batch(x) = reshape(x, (1,size(x)...))
batch(xs...) = vcat(map(batch, xs)...)

unbatch(xs) = reshape(xs, size(xs)[2:end])

type Model
  session::Session
  inputs::Vector{Tensor}
  graph::Tensor
  grad::Tensor
end

function tf(model)
  sess = Session(Graph())
  input = placeholder(Float32)
  g = graph(model, input)
  run(sess, initialize_all_variables())
  Model(sess, [input], g, gradients(g, input))
end

function (m::Model)(args...)
  @assert length(args) == length(m.inputs)
  unbatch(run(m.session, m.graph, Dict(zip(m.inputs, map(batch, args)))))
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
  Y = placeholder(Float32)
  Loss = loss(m.graph, Y)
  minimize_op = TensorFlow.train.minimize(opt, Loss)
  for e in 1:epoch
    info("Epoch $e\n")
    @progress for (x, y) in train
      y, cur_loss, _ = run(m.session, vcat(m.graph, Loss, minimize_op),
                           Dict(m.inputs[1]=>batch(x), Y=>batch(y)))
      if i % 5000 == 0
        @show y
        @show accuracy(m, test)
      end
      i += 1
    end
  end
end

type Op
  f
  shape
end

Op(f) = Op(f, (d...) -> nothing)

graph(op::Op, xs...) = op.f(xs...)
Flux.shape(op::Op, d...) = op.shape(d...)

end
