import Base: @get!
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

graph(p::MaxPool, x) =
  nn.max_pool(x, [1, p.size..., 1], [1, p.stride..., 1], "VALID")

graph(::Flow.Group, xs...) = (xs...,)

graph(params::Associative, c::Conv2D, x) =
  nn.conv2d(x, graph(params, c.filter), [1,c.stride...,1], "VALID")

type Op
  f
  shape
end

Op(f) = Op(f, (d...) -> nothing)

graph(op::Op, xs...) = op.f(xs...)
Flux.shape(op::Op, d...) = op.shape(d...)

graph{T<:AArray}(params::Associative, p::Flux.Param{T}) =
  @get!(params, p, Variable(p.x))

function graph(params::Associative, v::IVertex, args...)
  # TODO: check number of arguments
  v = spliceinputs(v, map(constant, args)...) |> detuple
  postwalk(v) do v
    vertex(graph(params, cvalue(v), cvalue.(inputs(v))...))
  end |> value
end

function graph(params::Associative, model, args...)
  g = Flux.graph(model)
  g == nothing && return graph(model, args...)
  graph(params, g, args...)
end

function tograph(model, args...)
  params = Dict{Flux.Param,Tensor}()
  g = graph(params, model, args...)
  return params, g
end

TensorFlow.Tensor(m::Flux.Model, args...) = graph(Dict(), m, args...)

RawTensor(data::Union{Batch,Seq}) = RawTensor(rawbatch(data))
