# This is hacky; we'll eventually reuse Cassette for better tracing.

using ..Flux.Tracker: Tracked, Broadcasted, param, tracker, istracked, isleaf
using DataFlow
using DataFlow: Call, Lambda, iscall, isconstant, prewalk, vertex, syntax,
  inputnode, constant

vcall(f, args...) = vertex(DataFlow.Call(), constant(f), args...)
vcall(f::Broadcasted, args...) = vcall(broadcast, constant(f.f), args...)

graph(x::Tracked, inputs...; cache = ObjectIdDict()) =
  vcall(x.f.func, map(x -> graph(x, inputs...; cache = cache), x.f.args)...)

function graph(x, inputs...; cache = ObjectIdDict())
  haskey(cache, x) && return cache[x]
  i = findfirst(inputs, x)
  cache[x] =
    i > 0 ? inputnode(i) :
    istracked(x) && !isleaf(x) ? graph(tracker(x), inputs...; cache = cache) :
    constant(x)
end

function trace(f, args...)
  inputs = param.(args)
  graph(f(inputs...), inputs...)
end

# Graph manipulation

function cacheall(v, buf = () -> UInt8[])
  prewalk(v) do v
    iscall(v) && isconstant(v[1]) || return v
    f = v[1].value.value
    return vertex(Call(), constant(Cached(f, buf())), v[2:end]...)
  end
end

function eval_func(v, n)
  v = vertex(Lambda(n, v))
  v |> syntax |> eval
end

function compile(f, args...)
  v = trace(f, args...)
  eval_func(cacheall(v), length(args))
end
