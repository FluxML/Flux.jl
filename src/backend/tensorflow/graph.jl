using Base: @get!
using DataFlow: Constant, constant, Split
using DataFlow.Interpreter
using Flux: imap
using TensorFlow: RawTensor, TFException

# TODO: implement Julia's type promotion rules

node(x::Tuple) = map(node, x)
node(x::Tensor) = x
node(x::Variable) = x
node(x::Number) = TensorFlow.constant(Float32(x))

graph(::typeof(tuple), args...) = (args...,)
graph(s::Split, t::Tuple) = t[s.n]
graph(::typeof(softmax), x) = nn.softmax(x)
graph(::typeof(relu), x) = nn.relu(x)
graph(::typeof(σ), x) = nn.sigmoid(x)
graph(::typeof(hcat), xs...) = concat(1, xs)
graph(::typeof(seq), xs, n) = TensorFlow.unpack(xs, num = n, axis = 1)

for op in (tanh, *, .*, .+, .-)
  @eval graph(::typeof($op), args...) = $op(args...)
end

graph(::typeof(.-), args...) = -(args...)

# reshape hack due to https://github.com/malmaud/TensorFlow.jl/issues/79
batchsize(x::Tensor) = reduce_sum(slice(TensorFlow.shape(x), [0], [1]))
graph(::typeof(flatten), x) = reshape(x, pack([batchsize(x), Int32(-1)]))
graph(r::Reshape, x) = reshape(x, pack([batchsize(x), map(Int32, r.dims)...]))

graph(::Input, x) = x

graph(p::MaxPool, x) =
  nn.max_pool(x, [1, p.size..., 1], [1, p.stride..., 1], "VALID")

graph(op::Op, xs...) = op.f(xs...)

function graph(ctx::Context, model, args...)
  node = graph(model, interpv(ctx, args)...)
  node isa Tensor && (ctx[:stacks][node.op.name] = stack(ctx))
  return node
end

interp(ctx, c::Conv2D, x) =
  nn.conv2d(interpv(ctx, x), interp(ctx, Constant(c.filter)), [1,c.stride...,1], "VALID")

interp{T<:AArray}(ctx, p::Constant{Flux.Param{T}}) =
  haskey(ctx[:params], p.value) ?
     ctx[:params][p.value] :
    (ctx[:params][p.value] = Variable(convertel(Float32, p.value.x)))

interp(ctx, p::Constant) = p.value

function interp(ctx, model, args...)
  g = Flux.graph(model)
  g == nothing && return graph(ctx, model, args...)
  DataFlow.iscyclic(g) && error("This model has a cycle; try unrolling it first.")
  interpret(ctx, g, interpv(ctx, args)...)
end

function tograph(model, args...)
  ctx = Context(mux(iline, ilambda, imap, interp),
                params = ObjectIdDict(), stacks = Dict())
  out = interp(ctx, model, map(constant, args)...)
  return ctx[:params], ctx[:stacks], out
end

TensorFlow.Tensor(m::Flux.Model, args...) = tograph(m, args...)[3]

RawTensor(data::Union{Batch,Seq}) = RawTensor(rawbatch(data))

# Error Handling

using Juno
using MacroTools: @q
using DataFlow.Interpreter: Exception, totrace
Juno.errmsg(e::TFException) = string(e.status)

function errnode(e::TFException)
  m = match(r"Node: ([\w\d]+) =", string(e.status))
  m == nothing && return
  m.captures[1]
end

errnode(e) = nothing

macro tferr(stk, ex)
  @q try
    $(esc(ex))
  catch e
    (node = errnode(e)) != nothing || rethrow()
    stk = $(esc(stk))
    haskey(stk, node) || rethrow()
    throw(Exception(e, totrace(stk[node])))
  end
end
