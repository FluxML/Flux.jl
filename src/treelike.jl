import Adapt: adapt

children(x) = ()
mapchildren(f, x) = x

children(x::Tuple) = x
mapchildren(f, x::Tuple) = map(f, x)

function treelike(T, fs = fieldnames(T))
  @eval current_module() begin
    Flux.children(x::$T) = ($([:(x.$f) for f in fs]...),)
    Flux.mapchildren(f, x::$T) = $T(f.($children(x))...)
  end
end

isleaf(x) = isempty(children(x))

function mapleaves(f, x; cache = ObjectIdDict())
  haskey(cache, x) && return cache[x]
  cache[x] = isleaf(x) ? f(x) : mapchildren(x -> mapleaves(f, x, cache = cache), x)
end

using DataFlow: OSet

function prefor(f, x; seen = OSet())
  x ∈ seen && return
  f(x)
  foreach(x -> prefor(f, x, seen = seen), children(x))
  return
end

function params(m)
  ps = []
  prefor(p ->
    Tracker.istracked(p) && Tracker.isleaf(p) &&
      !any(p′ -> p′ === p, ps) && push!(ps, p),
    m)
  return ps
end

params(m...) = params(m)

function loadparams!(m, xs)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copy!(data(p), data(x))
  end
end

# CPU/GPU movement conveniences

cpu(m) = mapleaves(x -> adapt(Array, x), m)

gpu_adaptor = identity

@require CuArrays begin
  global gpu_adaptor = CuArrays.cu
end

gpu(x) = mapleaves(gpu_adaptor, x)
