import Base: size

struct Tensor
  name
  size
end

size(t::Tensor) = t.size

uid = ((id)->()->id+=1)(0)

function <<(ctx::Context, str)
  ctx[:depth] <= ctx[:max_depth] && println(ctx[:buffer], "  " ^ ctx[:depth], str...)
  ctx
end

function <<(ctx::Context, t::Tensor)
  ctx[:memory] += isempty(t.size) ? 0 : *(t.size...)
  ctx
end

function scope(f, ctx, name)

end

function interp(ctx, c::DataFlow.Constant)
  interp(ctx, c.value)
end

function interp(ctx, c::Flux.Param)
  t = Tensor(uid(), size(c.x)) # TODO: is it possible to get variable name?
  ctx << t
  ctx << """$(t.name) [shape=box, label="param::$(t.size)"]"""
  t
end

function interp(ctx, a::AArray)
  t = Tensor(uid(), size(a))
  ctx << t
  ctx << """$(t.name) [shape=hexagon, label="input::$(t.size)"]"""
  t
end

interp(ctx, t::Tensor) = t

function interp(ctx, m::Flux.Model, xs...)
  g = Flux.graph(m)
  g == nothing && error("graphviz backend doesn't support $m")
  DataFlow.iscyclic(g) && error("This model has a cycle; try unrolling it first.")

  if ctx[:depth] < ctx[:max_depth]
    ctx << "subgraph cluster_$(uid()) {" << "  label = \"$(typeof(m))\""
  end

  ctx[:depth] += 1
  result = interpret(ctx, g, xs...)
  ctx[:depth] -= 1

  if ctx[:depth] < ctx[:max_depth]
    ctx << "}"
    result
  else
    t = Tensor(uid(), result.size)
    ctx << """$(t.name) [shape=diamond, label="$(typeof(m))::$(t.size)"]"""

    for x in xs
      ctx << "$(x.name) -> $(t.name)"
    end

    t
  end
end

interp(ctx, i::Input, t) = t

function interp(ctx, c::Conv2D, x)
  size_intered = Flux.infer(c, x.size)
  node = replace("""
    Conv2D(
      kernel: $(size(c.filter.x, 4))
      filter: $(join(size(c.filter.x)[1:2], 'x'))
      stride: $(join(c.stride, 'x'))
      border: valid
    )""", '\n', "\\n")
  t = Tensor(uid(), size_intered == nothing ? () : size_intered)
  ctx << t
  ctx << """$(t.name) [shape=oval, label="$node::$(t.size)"]"""
  ctx << "$(x.name) -> $(t.name)"
  t
end

function interp(ctx, p::MaxPool, x)
  size = Flux.infer(p, x.size)
  node = replace("""
    Pool(
      filter: $(join(p.size, 'x'))
      stride: $(join(p.stride, 'x'))
      function: max
    )""", '\n', "\\n")
  t = Tensor(uid(), size == nothing ? () : size)
  ctx << t
  ctx << """$(t.name) [shape=oval, label="$node::$(t.size)"]"""
  ctx << "$(x.name) -> $(t.name)"
  t
end

function interp(ctx, r::Reshape, x)
  t = Tensor(uid(), *(x.size[2:end]...) == *(r.dims...) ? (x.size[1], r.dims...) : ())
  ctx << """$(t.name) [shape=oval, label="Reshape::$(t.size)"]"""
  ctx << "$(x.name) -> $(t.name)"
  t
end

function interp(ctx, m::Flux.Stateful)
  error("graphviz backend doesn't support recurrent model yet")
end

interp(ctx, m::Flux.SeqModel) = interp(ctx, m.model)

for op in (:.+, :.-, :.*, :./)
  op_str = string(op)
  @eval function interp(ctx, ::typeof($op), xs...)
    t = Tensor(uid(), try
      any(isempty, map(size, xs)) ? () :
      Base.Broadcast.broadcast_shape(map(size, xs)...)
    catch e
      e isa DimensionMismatch ? () : rethrow()
    end)
    ctx << t
    ctx << """$(t.name) [shape=oval, label="$($op_str)::$(t.size)"]"""
    for x in xs
      ctx << "$(x.name) -> $(t.name)"
    end
    t
  end
end

for op in (:tanh, :σ, :relu, :softmax)
  op_str = string(op)
  @eval function interp(ctx, ::typeof($op), x)
    t = Tensor(uid(), x.size) # we don't count memory here as it will almost certanly be optimized out
    ctx << """$(t.name) [shape=oval, label="$($op_str)::$(t.size)"]"""
    ctx << "$(x.name) -> $(t.name)"
    t
  end
end

for op in (:tuple, )
  @eval interp(ctx, ::typeof($op), xs...) = $op(xs...)
end

function interp(ctx, ::typeof(*), A::Tensor, B::Tensor)
  t = Tensor(uid(), length(A.size) == length(B.size) == 2 &&
                    A.size[2] == B.size[1] ?
                    (A.size[1], B.size[2]) : ())
  ctx << t
  ctx << """$(t.name) [shape=oval, label="*::$(t.size)"]"""
  ctx << "$(A.name) -> $(t.name)"
  ctx << "$(B.name) -> $(t.name)"
  t
end

function interp(ctx, ::typeof(*), A::Tensor, x::Number)
  t = Tensor(uid(), A.size)
  ctx << """$(t.name) [shape=oval, label="(* $x)::$(t.size)"]"""
  ctx << "$(A.name) -> $(t.name)"
  t
end

interp(ctx, ::typeof(*), x::Number, A::Tensor) = interp(ctx, *, A, x)
