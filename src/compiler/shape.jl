using DataFlow.Interpreter

type Hint
  typ
end

DataFlow.tocall(h::Hint, x) = :($x::$(h.typ))

function gethint(v::IVertex)
  while isa(value(v), Union{Line,Frame}) v = v[1] end
  isa(value(v), Hint) && return value(v).typ
  return
end

ihint(f, ctx::Context, h::Hint, x) = vertex(h, x)
ihint(f, args...) = f(args...)

hintify(ctx, c::Constant) = hintify(ctx, state(c.value))
hintify(ctx, xs::AbstractArray) = vertex(Hint(size(xs)), constant(:_))

interpshape = mux(ilinev, ihint, iargs, ituple, hintify)

function hintify(ctx, f, xs...)
  sh = infer(f, map(gethint, xs)...)
  sh ≠ nothing ? vertex(Hint(sh), vertex(f, xs...)) :
  !any(x->x==nothing, xs) && graph(f) ≠ nothing ? interpret(Context(interpshape), graph(f), xs...) :
    vertex(f, xs...)
end

function shapesv(f, args...)
  (g = graph(f)) == nothing && return
  ins = [vertex(Hint(d), inputnode(i)) for (i,d) in enumerate(args)]
  interpret(Context(interpshape), g, ins...)
end

shapes(args...) = shapesv(args...) |> syntax |> applylines |> (x->prettify(x, lines=true))

# Inference primitives

infer(f, args...) = graph(f) == nothing ? nothing : gethint(shapesv(f, args...))

function infer(::typeof(*), a::NTuple{2}, b::NTuple{2})
  a[2] == b[1] || return nothing
  (a[1], b[2])
end

# TODO: make correct
infer(::typeof(+), a, b) = a
