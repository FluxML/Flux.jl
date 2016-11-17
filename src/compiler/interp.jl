using DataFlow: interpret, interpret, interptuple, interplambda, interpconst, Context

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

function interpmap(cb)
  function interp(ctx, ::typeof(map), f, xs...)
    f, xs = interpret(ctx, (f, xs))
    xs′ = astuples(xs)
    xs′ ≠ nothing ?
      group(map(f, xs′...)...) :
      cb(ctx, map, constant(f), xs...)
  end
  interp(args...) = cb(args...)
end

function interp(ctx, model, xs...)
  g = graph(model)
  g == nothing && return vertex(model, map(constant, interpret(ctx, xs))...)
  interpret(ctx, g, xs...)
end

expand(graph, xs...) =
  interp(Context(interplambda(interpmap(interpconst(interptuple(interp))))), graph, xs...)
