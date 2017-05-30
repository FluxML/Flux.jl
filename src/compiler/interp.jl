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

function interp(ctx, f, xs...)
  g = graph(f)
  @icatch(ctx, g ≠ nothing ?
    interpret(ctx, reifyparams(g), xs...) :
    f(xs...))
end

interp(ctx::Context, c::Constant{<:Param}) = c.value.x
interp(ctx::Context, c::Constant) = c.value

function interpmodel(m, args...)
  ctx = Context(mux(iline, ilambda, iargs, ituple, interp))
  @ithrow interp(ctx, m, args...)
end
