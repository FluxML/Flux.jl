using DataFlow.Interpreter

function astuple(xs::Vertex)
  isconstant(xs) && isa(value(xs).value, Tuple) ? value(xs).value :
  isa(xs, Vertex) && value(xs) == tuple ? inputs(xs) :
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
  @ithrow(ctx, g ≠ nothing ?
    interpret(ctx, reifyparams(g), xs...) :
    f(xs...))
end

# TODO: batching should be secondary

function interpmodel(m, args::Batch...)
  ctx = Context(mux(iline, ilambda, iconst, iargs, ituple, interp))
  rebatch(@icatch interp(ctx, m, map(rawbatch, args)...))
end

interpmodel(m, args...) = unbatchone(interpmodel(m, batchone(args)...))
