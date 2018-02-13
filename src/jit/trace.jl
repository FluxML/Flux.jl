# This is hacky; we'll eventually reuse Cassette for better tracing.

using ..Flux.Tracker: Tracked, Broadcasted, param, tracker, istracked, isleaf
using DataFlow
using DataFlow: inputnode, constant

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
