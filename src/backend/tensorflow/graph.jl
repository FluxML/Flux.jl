import Flow: Constant, postwalk, value, inputs, constant
import TensorFlow: RawTensor

cvalue(x) = x
cvalue(c::Constant) = c.value
cvalue(v::Vertex) = cvalue(value(v))

graph(x::Tensor) = x

graph(::typeof(*), args...) = *(args...)
graph(::typeof(+), args...) = +(args...)
graph(::typeof(softmax), x) = nn.softmax(x)
graph(::typeof(relu), x) = nn.relu(x)
graph(::typeof(tanh), x) = tanh(x)
graph(::typeof(σ), x) = nn.sigmoid(x)

# reshape hack due to https://github.com/malmaud/TensorFlow.jl/issues/79
batchsize(x::Tensor) = reduce_sum(slice(TensorFlow.shape(x), [0], [1]))
graph(::typeof(flatten), x) = reshape(x, pack([batchsize(x), Int32(-1)]))
graph(r::Reshape, x) = reshape(x, pack([batchsize(x), map(Int32, r.dims)...]))

graph(::Input, x) = x

graph(c::Conv2D, x) =
  nn.conv2d(x, graph(c.filter), [1,c.stride...,1], "VALID")

graph(p::MaxPool, x) =
  nn.max_pool(x, [1, p.size..., 1], [1, p.stride..., 1], "VALID")

graph(::Flow.Group, xs...) = (xs...,)

type Op
  f
  shape
end

Op(f) = Op(f, (d...) -> nothing)

graph(op::Op, xs...) = op.f(xs...)
Flux.shape(op::Op, d...) = op.shape(d...)

# TODO: detect variable reuse
graph{T<:AArray}(p::Flux.Param{T}) = Variable(p.x)

function graph(model::Model, args...)
  g = Flux.graph(model)
  g ≠ nothing || error("No graph for $model")
  g = spliceinputs(g, map(constant, args)...) |> detuple
  postwalk(g) do v
    vertex(graph(cvalue(v), cvalue.(inputs(v))...))
  end |> value
end

TensorFlow.Tensor(m::Flux.Model, args...) = graph(m, args...)

RawTensor(data::Union{Batch,Seq}) = RawTensor(rawbatch(data))
