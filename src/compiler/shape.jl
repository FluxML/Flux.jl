export @shapes

Dims{N} = NTuple{N,Int}

struct Hint
  typ
end

DataFlow.tocall(h::Hint, x) = :($x::$(h.typ))

arghint(p::Param) = arghint(state(p))
arghint(xs::AbstractArray) = vertex(Hint(size(xs)), constant(:_))
arghint(x) = constant(x)

function gethint(v::IVertex)
  while value(v) isa Union{Line,Frame} v = v[1] end
  value(v) isa Hint && return value(v).typ
  return
end

ihint(f, ctx::Context, h::Hint, x) = vertex(h, x)
ihint(f, args...) = f(args...)

function hintify(ctx, f, xs...)
  xs = arghint.(xs)
  sh = infer(f, map(gethint, xs)...)
  sh ≠ nothing ? vertex(Hint(sh), vertex(f, xs...)) :
  !any(x->x==nothing, xs) && graph(f) ≠ nothing ? interpret(Context(interpshape), graph(f), xs...) :
    vertex(f, xs...)
end

interpshape = mux(ilinev, iconst, ihint, iargs, hintify)

function shapesv(f, args...)
  (g = graph(f)) == nothing && return
  ins = [vertex(Hint(d), inputnode(i)) for (i,d) in enumerate(args)]
  interpv(Context(interpshape), detuple(spliceinputs(g, ins...)))
end

shapes(args...) = shapesv(args...) |> syntax |> applylines |> (x->prettify(x, lines=true))

# Inference primitives

infer(f, args...) = graph(f) == nothing ? nothing : gethint(shapesv(f, args...))

infer(::typeof(tuple), xs...) = (xs...,)
infer(s::Split, xs::Tuple) = 1 ≤ s.n ≤ length(xs) ? xs[s.n] : nothing
infer(::typeof(identity), x) = x

function infer(::typeof(*), a::Dims{2}, b::Dims{2})
  a[2] == b[1] || return nothing
  (a[1], b[2])
end

infer(::typeof(broadcast), f, xs::Dims...) = Base.Broadcast.broadcast_shape(xs...)
# Old broadcast versions
infer(::typeof(.+), xs::Dims...) = Base.Broadcast.broadcast_shape(xs...)

# Shapes macro

macro shapes(ex)
  @capture(ex, f_(args__)) || error("@shapes f(args...)")
  :(shapes($(esc(f)), mapt(size, ($(map(esc, args)...),))...))
end

# Shim for kicking off shape inference

export Input

struct Input{N}
  dims::Dims{N}
end

Input(i...) = Input((i...,))

(::Input)(x) = x

inferchain(f::Input, xs) = (-1, f.dims...)
