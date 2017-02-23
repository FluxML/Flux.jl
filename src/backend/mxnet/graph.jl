function nodename(s::mx.SymbolicNode)
  name = Ref{mx.char_p}(0)
  success = Ref(0)
  mx.@mxcall(:MXSymbolGetName, (mx.MX_handle, Ref{mx.char_p}, Ref{Int}), s.handle.value, name, success)
  @assert success[] != -1
  return Symbol(unsafe_wrap(String, name[]))
end

using Base: @get!
using DataFlow: Constant, constant
using DataFlow.Interpreter
using DataFlow.Interpreter: Exception, totrace
using Flux: imap

# TODO: implement Julia's type promotion rules

node(x::Tuple) = map(node, x)
node(x::mx.SymbolicNode) = x

graph(::typeof(tuple), args...) = (args...,)
graph(::typeof(+), args...) = mx.broadcast_plus(args...)
graph(::typeof(*), x, W) = mx.dot(W, x) # Adjustments for batching
graph(::typeof(σ), x) = mx.Activation(x, act_type = :sigmoid)
graph(::typeof(relu), x) = mx.Activation(x, act_type = :relu)
graph(::typeof(tanh), x) = mx.Activation(x, act_type = :tanh)
graph(::typeof(flatten), x) = mx.Flatten(x)

graph(::typeof(softmax), xs) =
  mx.broadcast_div(exp(xs), mx.Reshape(mx.sum(exp(xs)), shape = (1,1)))

graph(::typeof(cat), dim::Integer, a...) = mx.Concat(a..., dim = dim)
graph(::typeof(vcat), a...) = graph(cat, 1, a...)

graph(::Input, x) = x

graph(ctx::Context, d::Affine, x) =
  !ctx[:feedforward] ? invoke(graph, (Context, Any, typeof(x)), ctx, d, x) :
    register(ctx,
      mx.FullyConnected(x,
                        num_hidden = size(d.W.x, 2),
                        weight = var(ctx, d.W, size(d.W)),
                        bias = var(ctx, d.b, size(d.b, 2))))

# TODO: use actual params}
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

function var(ctx::Context, p::Flux.Param, size = nothing)
  id = gensym()
  ctx[:params][id] = size == nothing ? rebatch_last(p.x) : reshape(p.x, size...)
  return mx.Variable(id)
end

graph{T<:AArray}(ctx::Context, p::Constant{Flux.Param{T}}) = var(ctx, p.value)

graph(ctx::Context, p::Constant) = node(p.value)

function graph(ctx::Context, model, args...)
  g = Flux.graph(model)
  g == nothing && return register(ctx, @icatch ctx graph(model, args...))
  DataFlow.iscyclic(g) && error("This model has a cycle; try unrolling it first.")
  interpret(ctx, g, args...)
end

graph′(ctx::Context, args...) = @icatch ctx graph(ctx, args...)

function tograph(model, args...; feedforward = false)
  ctx = Context(mux(iline, ilambda, imap, iargs, ituple, graph′),
                params = Dict(), stacks = Dict(),
                feedforward = feedforward)
  out = @ithrow graph(ctx, model, args...)
  return ctx[:params], ctx[:stacks], out
end

# Error Handling

using Juno
Juno.errmsg(e::mx.MXError) = e.msg

function errnode(e::mx.MXError)
  m = match(r"Error in operator (\w+)", e.msg)
  m == nothing && return
  Symbol(m.captures[1])
end

macro mxerr(stk, ex)
  :(try
      $(esc(ex))
    catch e
      (isa(e, mx.MXError) && (node = errnode(e)) != nothing) || rethrow()
      stk = $(esc(stk))
      throw(Exception(e, totrace(stk[node])))
    end)
end
