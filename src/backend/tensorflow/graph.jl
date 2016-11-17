using Base: @get!
using DataFlow: Constant, constant, Context, interpret, Split, interptuple, interplambda, interpconst
using Flux: interpmap
using TensorFlow: RawTensor

# TODO: implement Julia's type promotion rules

node(x::Tuple) = map(node, x)
node(x::Tensor) = x
node(x::Number) = TensorFlow.constant(Float32(x))

graph(::typeof(tuple), args...) = (args...,)
graph(s::Split, t::Tuple) = t[s.n]
graph(::typeof(softmax), x) = nn.softmax(x)
graph(::typeof(relu), x) = nn.relu(x)
graph(::typeof(σ), x) = nn.sigmoid(x)
graph(::typeof(hcat), xs...) = concat(1, xs)
graph(::typeof(seq), xs, n) = TensorFlow.unpack(xs, num = n, axis = 1)
graph(::typeof(.+), args...) = +(args...)

for op in (tanh, *, .*, +, -, .-)
  @eval graph(::typeof($op), args...) = $op(args...)
end

# reshape hack due to https://github.com/malmaud/TensorFlow.jl/issues/79
batchsize(x::Tensor) = reduce_sum(slice(TensorFlow.shape(x), [0], [1]))
graph(::typeof(flatten), x) = reshape(x, pack([batchsize(x), Int32(-1)]))
graph(r::Reshape, x) = reshape(x, pack([batchsize(x), map(Int32, r.dims)...]))

graph(::Input, x) = x

graph(p::MaxPool, x) =
  nn.max_pool(x, [1, p.size..., 1], [1, p.stride..., 1], "VALID")

graph(op::Op, xs...) = op.f(xs...)

interp(ctx, c::Conv2D, x) =
  nn.conv2d(interpret(ctx, x), interp(ctx, Constant(c.filter)), [1,c.stride...,1], "VALID")

interp{T<:AArray}(ctx, p::Constant{Flux.Param{T}}) =
  haskey(ctx[:params], p.value) ?
     ctx[:params][p.value] :
    (ctx[:params][p.value] = Variable(p.value.x))

function interp(ctx, model, args...)
  g = Flux.graph(model)
  g == nothing && return graph(model, interpret(ctx, args)...)
  DataFlow.iscyclic(g) && error("This model has a cycle; try unrolling it first.")
  interpret(ctx, g, interpret(ctx, args)...)
end

function tograph(model, args...)
  ctx = Context(interplambda(interptuple(interpmap(interpconst(interp)))), params = ObjectIdDict())
  out = interp(ctx, model, map(constant, args)...)
  return ctx[:params], out
end

TensorFlow.Tensor(m::Flux.Model, args...) = tograph(m, args...)[2]

RawTensor(data::Union{Batch,Seq}) = RawTensor(rawbatch(data))

function makesession(model, n)
  sess = Session(Graph())
  inputs = [placeholder(Float32) for _ = 1:n]
  params, output = tograph(model, inputs...)
  run(sess, initialize_all_variables())
  sess, params, inputs, output
end

function storeparams!(sess, params)
  for (p, t) in params
    p.x = run(sess, t)
  end
end
