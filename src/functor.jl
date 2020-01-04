import Adapt: adapt, adapt_storage
using Zygote: IdSet

functor(x) = (), _ -> x

functor(x::Tuple) = x, y -> y
functor(x::NamedTuple) = x, y -> y

functor(x::AbstractArray) = x, y -> y
functor(x::AbstractArray{<:Number}) = (), _ -> x

function makefunctor(m::Module, T, fs = fieldnames(T))
  @eval m begin
    Flux.functor(x::$T) = ($([:($f=x.$f) for f in fs]...),), y -> $T(y...)
  end
end

function functorm(T, fs = nothing)
  fs == nothing || isexpr(fs, :tuple) || error("@functor T (a, b)")
  fs = fs == nothing ? [] : [:($(map(QuoteNode, fs.args)...),)]
  :(makefunctor(@__MODULE__, $(esc(T)), $(fs...)))
end

macro functor(args...)
  functorm(args...)
end

isleaf(x) = functor(x)[1] === ()

function fmap1(f, x)
  func, re = functor(x)
  re(map(f, func))
end

function fmap(f, x; cache = IdDict())
  haskey(cache, x) && return cache[x]
  cache[x] = isleaf(x) ? f(x) : fmap1(x -> fmap(f, x, cache = cache), x)
end

trainable(m) = functor(m)[1]

params!(p::Params, x::AbstractArray{<:Number}, seen = IdSet()) = push!(p, x)

function params!(p::Params, x, seen = IdSet())
  x in seen && return
  push!(seen, x)
  for child in trainable(x)
    params!(p, child, seen)
  end
end

function params(m...)
  ps = Params()
  params!(ps, m)
  return ps
end

# Deprecated stuff
macro treelike(args...)
  functorm(args...)
end
mapleaves(f, x) = fmap(f, x)

function loadparams!(m, xs)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end

# CPU/GPU movement conveniences

cpu(m) = fmap(x -> adapt(Array, x), m)

gpu(x) = use_cuda[] ? fmap(CuArrays.cu, x) : x

# Precision

adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Real}) = convert.(T, xs)

paramtype(T::Type{<:Real}, m) = fmap(x -> adapt(T, x), m)

f32(m) = paramtype(Float32, m)
f64(m) = paramtype(Float64, m)
