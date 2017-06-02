function nodename(s::mx.SymbolicNode)
  name = Ref{mx.char_p}(0)
  success = Ref(0)
  mx.@mxcall(:MXSymbolGetName, (mx.MX_handle, Ref{mx.char_p}, Ref{Int}), s.handle.value, name, success)
  @assert success[] != -1
  return Symbol(unsafe_string(name[]))
end

using Base: @get!
using DataFlow: Constant, constant
using DataFlow.Interpreter
using DataFlow.Interpreter: Exception, totrace
import Flux: mapt, broadcastto, ∘

# TODO: implement Julia's type promotion rules

node(x::Tuple) = map(node, x)
node(x::mx.SymbolicNode) = x

graph(::typeof(tuple), args...) = (args...,)
graph(::typeof(identity), x) = x
graph(::typeof(*), xs...) = mx.dot(reverse(xs)...) # Work around MXNet shape hack
graph(::typeof(σ), x) = mx.Activation(x, act_type = :sigmoid)
graph(::typeof(relu), x) = mx.Activation(x, act_type = :relu)
graph(::typeof(tanh), x) = mx.Activation(x, act_type = :tanh)
graph(::typeof(flatten), x) = mx.Flatten(x)
graph(::typeof(hcat), xs...) = mx.concat(xs..., dim = 2-1)
graph(::typeof(vec), xs) = reshape(xs, shape = (-1,))

graph(::typeof(broadcast), ::typeof(+), args...) = mx.broadcast_plus(args...)
graph(::typeof(broadcast), ::typeof(-), args...) = mx.broadcast_sub(args...)
graph(::typeof(broadcast), ::typeof(*), args...) = mx.broadcast_mul(args...)
graph(::typeof(broadcast), ::typeof(/), args...) = mx.broadcast_div(args...)
graph(::typeof(broadcastto), xs, shape) = mx.broadcast_to(xs, shape = map(i -> i≤1?0:i, reverse(shape)))
# Old broadcasters
graph(::typeof(broadcast), ::typeof(exp), xs) = exp(xs)
graph(::typeof(.+), args...) = mx.broadcast_plus(args...)
graph(::typeof(.*), args...) = mx.broadcast_mul(args...)
graph(::typeof(.-), args...) = mx.broadcast_sub(args...)

graph(::typeof(softmax), xs) =
  mx.broadcast_div(exp(xs), mx.sum(exp(xs), axis = 1, keepdims=true))

graph(::typeof(cat), dim::Integer, a...) = mx.Concat(a..., dim = dim)
graph(::typeof(vcat), a...) = graph(cat, 1, a...)

graph(::typeof(map), f, xss::Tuple...) = map(f, xss...)
graph(::typeof(getindex), t::Tuple, n::Integer) = t[n]
graph(::typeof(sum), xs::Tuple) = reduce((a, b) -> graph(broadcast, +, a, b), xs)

a::mx.SymbolicNode ∘ b::mx.SymbolicNode = mx.broadcast_mul(a, b)

graph(::Input, x) = x

struct AlterParam
  param
  load
  store
end

Base.size(p::AlterParam) = size(p.load(p.param.x))
Base.copy!(xs, p::AlterParam) = copy!(xs, p.load(p.param.x))

graph(ctx::Context, d::Affine, x) =
  !ctx[:feedforward] ? invoke(graph, Tuple{Context, Any, typeof(x)}, ctx, d, x) :
    register(ctx,
      mx.FullyConnected(mx.SymbolicNode, data = x,
                        num_hidden = size(d.W.x, 2),
                        weight = var(ctx, AlterParam(d.W, x->x', nothing)),
                        bias = var(ctx, AlterParam(d.b, x->squeeze(x, 1), nothing))))

# TODO: use actual params
graph(ctx::Context, c::Conv2D, x) =
  mx.Convolution(x,
                 kernel = size(c.filter, 1, 2),
                 num_filter = size(c.filter, 4),
                 stride = c.stride)

graph(ctx::Context, p::MaxPool, x) =
  mx.Pooling(x,
             pool_type = :max,
             kernel = p.size,
             stride = p.stride)

function register(ctx::Context, node::mx.SymbolicNode)
  ctx[:stacks][nodename(node)] = stack(ctx)
  return node
end

register(ctx::Context, node) = node

function var(ctx::Context, p::Union{Flux.Param{<:AArray},AArray,AlterParam})
  id = gensym()
  ctx[:params][id] = p
  return mx.Variable(id)
end

var(ctx::Context, x) = x

function graph(ctx::Context, model, args...)
  args = var.(ctx, args)
  g = Flux.graph(model)
  g == nothing && return register(ctx, @icatch ctx graph(model, args...))
  DataFlow.iscyclic(g) && error("This model has a cycle; try unrolling it first.")
  interpret(ctx, g, args...)
end

graph′(ctx::Context, args...) = @icatch ctx graph(ctx, args...)

function tograph(model, args...; feedforward = false)
  ctx = Context(mux(iline, iconst, ilambda, iargs, ituple, graph′),
                params = Dict(), stacks = Dict(),
                feedforward = feedforward)
  out = @ithrow graph(ctx, model, mapt(mx.Variable, args)...)
  return Graph(args, out, ctx[:params], ctx[:stacks])
end

# Error Handling

using Juno
using MacroTools: @q
Juno.errmsg(e::mx.MXError) = e.msg

function errnode(e::mx.MXError)
  m = match(r"Error in operator (\w+)", e.msg)
  m == nothing && return
  Symbol(m.captures[1])
end

striptrace(e::mx.MXError) = mx.MXError(split(e.msg, "\n")[1])

macro mxerr(stk, ex)
  @q try
    $(esc(ex))
  catch e
    (e isa mx.MXError && (node = errnode(e)) != nothing) || rethrow()
    stk = $(esc(stk))
    haskey(stk, node) || rethrow()
    throw(Exception(striptrace(e), totrace(stk[node])))
  end
end
