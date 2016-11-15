using DataFlow: interpret, interpret, interptuple, interplambda, interpconst, Context

function astuple(xs)
  isconstant(xs) && isa(value(xs).value, Tuple) ? value(xs).value :
  isa(xs, Vertex) && isa(value(xs), Group) ? inputs(xs) :
  nothing
end

function astuples(xs)
  xs = [astuple(x) for x in xs]
  all(x->!(x==nothing), xs) ? xs : nothing
end

# function interp(ctx, ::typeof(map), f, xs...)
#   f = interpret(ctx, f)
#   xs = interpret(ctx, xs)
#   xs′ = astuples(xs)
#   xs′ == nothing ? constant(:MAPFAIL) : @show map(f, xs...)
# end

function interp(ctx, model, xs...)
  @show model
  g = graph(model)
  g == nothing && return vertex(model, map(constant, interpret(ctx, xs))...)
  interpret(ctx, g, xs...)
end

expand(graph, xs...) =
  interp(Context(interplambda(interpconst(interptuple(interp)))), graph, xs...)
