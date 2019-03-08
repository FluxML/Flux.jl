import Adapt: adapt, adapt_storage
import .Zygote: IdSet

children(x) = ()
mapchildren(f, x) = x

children(x::Tuple) = x
children(x::NamedTuple) = x
mapchildren(f, x::Tuple) = map(f, x)
mapchildren(f, x::NamedTuple) = map(f, x)

function treelike(m::Module, T, fs = fieldnames(T))
  @eval m begin
    Flux.children(x::$T) = ($([:(x.$f) for f in fs]...),)
    Flux.mapchildren(f, x::$T) = $T(f.($children(x))...)
  end
end

macro treelike(T, fs = nothing)
  fs == nothing || isexpr(fs, :tuple) || error("@treelike T (a, b)")
  fs = fs == nothing ? [] : [:($(map(QuoteNode, fs.args)...),)]
  :(treelike(@__MODULE__, $(esc(T)), $(fs...)))
end

isleaf(x) = isempty(children(x))

function mapleaves(f, x; cache = IdDict())
  haskey(cache, x) && return cache[x]
  cache[x] = isleaf(x) ? f(x) : mapchildren(x -> mapleaves(f, x, cache = cache), x)
end

function prefor(f, x; seen = IdSet())
  x ∈ seen && return
  f(x)
  foreach(x -> prefor(f, x, seen = seen), children(x))
  return
end

function params(m)
  ps = Params()
  prefor(p ->
    p isa AbstractArray{<:Real} &&
      !any(p′ -> p′ === p, ps) && push!(ps, p),
    m)
  return ps
end

params(m...) = params(m)

function loadparams!(m, xs)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(data(p), data(x))
  end
end

# CPU/GPU movement conveniences

cpu(m) = mapleaves(x -> adapt(Array, x), m)

gpu_adaptor = identity

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  global gpu_adaptor = CuArrays.cu
end

gpu(x) = mapleaves(gpu_adaptor, x)

# Precision

adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Real}) = convert.(T, xs)

paramtype(T::Type{<:Real}, m) = mapleaves(x -> adapt(T, x), m)

f32(m) = paramtype(Float32, m)
f64(m) = paramtype(Float64, m)

# General parameter map

function mapparams(f, m)
  mapleaves(m) do x
    x isa Union{AbstractArray,Number} ? f(x) : x
  end
end
