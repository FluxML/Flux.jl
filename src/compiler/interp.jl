using DataFlow.Interpreter

function astuple(xs::Vertex)
  isconstant(xs) && value(xs).value isa Tuple ? value(xs).value :
  xs isa Vertex && value(xs) == tuple ? inputs(xs) :
  nothing
end

astuple(xs::Tuple) = xs

astuple(xs) = nothing

function astuples(xs)
  xs = [astuple(x) for x in xs]
  all(x->!(x==nothing), xs) ? xs : nothing
end

function imap(cb, ctx, ::typeof(map), f, xs...)
  f, xs = interpv(ctx, (f, xs))
  xs′ = astuples(xs)
  xs′ ≠ nothing ?
    group(map(f, xs′...)...) :
    cb(ctx, map, constant(f), xs...)
end

imap(f, args...) = f(args...)

function interp(ctx, f, xs...)
  g = graph(f)
  @icatch(ctx, g ≠ nothing ?
    interpret(ctx, reifyparams(g), xs...) :
    f(xs...))
end

# TODO: batching should be secondary

function interpmodel_(m, args...)
  ctx = Context(mux(iline, ilambda, iconst, iargs, ituple, interp))
  interp(ctx, m, args...)
end

interpmodel(m, args...) = @ithrow runrawbatched((xs...) -> interpmodel_(m, xs...), args...)
