# This is hacky; we'll eventually reuse Cassette for better tracing.

using ..Tracker, DataFlow
using ..Tracker: Tracked, Broadcasted, param, tracker, istracked, isleaf
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

function liftparams(v)
  ps = []
  v = prewalk(DataFlow.bumpinputs(v)) do v
    isconstant(v) && istracked(v.value.value) || return v
    push!(ps, v.value.value)
    DataFlow.vcall(getindex, inputnode(1), length(ps))
  end
  return v, ps
end

function cacheall(v, buf = () -> UInt8[])
  prewalk(v) do v
    iscall(v) && isconstant(v[1]) || return v
    f = v[1].value.value
    return vertex(Call(), constant(Cached(f, buf())), v[2:end]...)
  end
end

code(v, n) = syntax(vertex(Lambda(n, v)))

struct Compiled{F,T<:Tuple}
  model
  func::F
  params::T
end

(c::Compiled)(args...) =
  Tracker.track(Tracker.Call(c, args...),
                c.func(Tracker.data.(c.params), args...))

Base.show(io::IO, c::Compiled) = print(io, "Compiled(", c.model, ")")

function compile(f, args...)
  v = trace(f, args...)
  v, ps = liftparams(cacheall(v))
  Compiled(f, eval(code(v, length(args)+1)), (ps...,))
end

function source(f, args...)
  v = trace(f, args...)
  v, ps = liftparams(v)
  code(v, length(args)+1) |> prettify
end
