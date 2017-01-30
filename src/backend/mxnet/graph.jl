function nodename(s::mx.SymbolicNode)
  name = Ref{mx.char_p}(0)
  success = Ref(0)
  mx.@mxcall(:MXSymbolGetName, (mx.MX_handle, Ref{mx.char_p}, Ref{Int}), s.handle.value, name, success)
  @assert success[] != -1
  return Symbol(unsafe_wrap(String, name[]))
end

using Base: @get!
using DataFlow: Constant, constant, Context, interpret, Split,
  interpv, ituple, ilambda, iconst, iline, iargs, stack, mux
using Flux: imap

# TODO: implement Julia's type promotion rules

node(x::Tuple) = map(node, x)
node(x::mx.SymbolicNode) = x
# node(x::Number) = TensorFlow.constant(Float32(x))

graph(::typeof(tuple), args...) = (args...,)
graph(::typeof(*), args...) = mx.dot(args...)
graph(::typeof(+), args...) = mx.broadcast_plus(args...)
graph(::typeof(σ), x) = mx.Activation(data = x, act_type = :sigmoid)
graph(::typeof(relu), x) = mx.Activation(data = x, act_type=:relu)
graph(::typeof(tanh), x) = mx.Activation(data = x, act_type=:tanh)
graph(::typeof(flatten), x) = mx.Flatten(data = x)

graph(::typeof(softmax), xs) =
  mx.broadcast_div(exp(xs), mx.Reshape(mx.sum(exp(xs)), shape = (1,1)))

graph(::typeof(cat), dim::Integer, a...) = mx.Concat(a..., dim = dim)
graph(::typeof(vcat), a...) = node(cat, 1, a...)

graph(::Input, x) = x

# graph(vars, c::Conv, x) =
#   mx.Convolution(data = x,
#                  kernel = c.size,
#                  num_filter = c.features,
#                  stride = c.stride)
#
# graph(vars, p::MaxPool, x) =
#   mx.Pooling(data = x,
#              pool_type = :max,
#              kernel = p.size,
#              stride = p.stride)
#
# graph(vars, d::Dense, x) =
#   mx.FullyConnected(data = x,
#                     num_hidden = size(d.W.x, 1),
#                     weight = graph(vars, d.W),
#                     bias = graph(vars, d.b))

function interp{T<:AArray}(ctx, p::Constant{Flux.Param{T}})
  id = gensym()
  ctx[:params][id] = p.value.x
  return mx.Variable(id)
end

interp(ctx, p::Constant) = node(p.value)

function register(ctx::Context, node::mx.SymbolicNode)
  ctx[:stacks][nodename(node)] = stack(ctx)
  return node
end

register(ctx::Context, node) = node

function graph(ctx::Context, model, args...)
  register(ctx, graph(model, args...))
end

function interp(ctx, model, args...)
  g = Flux.graph(model)
  g == nothing && return graph(ctx, model, args...)
  DataFlow.iscyclic(g) && error("This model has a cycle; try unrolling it first.")
  interpret(ctx, g, args...)
end

function tograph(model, args...)
  ctx = Context(mux(iline, ilambda, imap, iargs, ituple, interp),
                params = Dict(), stacks = Dict())
  out = interp(ctx, model, map(constant, args)...)
  return ctx[:params], ctx[:stacks], out
end

# Error Handling

using Juno
Juno.errmsg(e::mx.MXError) = e.msg

function errnode(e::mx.MXError)
  m = match(r"Error in (\w+):", e.msg)
  m == nothing && return
  Symbol(m.captures[1])
end

macro mxerr(stk, ex)
  :(try
      $(esc(ex))
    catch e
      (isa(e, mx.MXError) && (node = errnode(e)) != nothing) || rethrow()
      stk = $(esc(stk))
      throw(DataFlow.Exception(e, DataFlow.totrace(stk[node])))
    end)
end
