function astuple(xs::Vertex)
  isconstant(xs) && value(xs[1]) isa Tuple ? value(xs[1]) :
  xs isa Vertex && value(xs) == tuple ? inputs(xs) :
  nothing
end

astuple(xs::Tuple) = xs

astuple(xs) = nothing

function astuples(xs)
  xs = [astuple(x) for x in xs]
  all(x->!(x==nothing), xs) ? xs : nothing
end

function interp(ctx, f, xs...)
  g = graph(f)
  g ≠ nothing && iscyclic(g) && error("Can't interpret cyclic graph")
  @icatch(ctx, g ≠ nothing ?
    interpret(ctx, g, xs...) :
    f(xs...))
end

function interpmodel(m, args...)
  ctx = Context(mux(iconst, iline, ilambda, iargs, ituple, interp))
  @ithrow interp(ctx, m, args...)
end

# Anonymous models

struct Capacitor
  graph::IVertex{Any}
end

(m::Capacitor)(xs...) = interpmodel(m, xs...)

graph(cap::Capacitor) = cap.graph

Base.show(io::IO, ::Capacitor) = print(io, "Capacitor(...)")
