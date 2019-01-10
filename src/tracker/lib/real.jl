struct TrackedReal{T<:Real} <: Real
  data::T
  tracker::Tracked{T}
end

TrackedReal(x::Real) = TrackedReal(x, Tracked{typeof(x)}(Call(), zero(x)))

data(x::TrackedReal) = x.data
tracker(x::TrackedReal) = x.tracker

track(f::Call, x::Real) = TrackedReal(x, Tracked{typeof(x)}(f, zero(x)))

function back!(x::TrackedReal; once = true)
    isinf(x) && error("Loss is Inf")
    isnan(x) && error("Loss is NaN")
    return back!(x, 1, once = once)
end

function Base.show(io::IO, x::TrackedReal)
  T = get(io, :typeinfo, Any)
  show(io, data(x))
  T <: TrackedReal || print(io, " (tracked)")
end

Base.decompose(x::TrackedReal) = Base.decompose(data(x))

Base.copy(x::TrackedReal) = x

Base.convert(::Type{TrackedReal{T}}, x::TrackedReal{T}) where T = x

Base.convert(::Type{TrackedReal{T}}, x::Real) where T = TrackedReal(convert(T, x))

Base.convert(::Type{TrackedReal{T}}, x::TrackedReal{S}) where {T,S} =
  error("Not implemented: convert tracked $S to tracked $T")

for op in [:(==), :≈, :<, :(<=)]
  @eval Base.$op(x::TrackedReal, y::Real) = Base.$op(data(x), y)
  @eval Base.$op(x::Real, y::TrackedReal) = Base.$op(x, data(y))
  @eval Base.$op(x::TrackedReal, y::TrackedReal) = Base.$op(data(x), data(y))
end

Base.eps(x::TrackedReal) = eps(data(x))
Base.eps(::Type{TrackedReal{T}}) where T = eps(T)

for f in :[isinf, isnan, isfinite].args
  @eval Base.$f(x::TrackedReal) = Base.$f(data(x))
end

Base.Printf.fix_dec(x::TrackedReal, n::Int, a...) = Base.Printf.fix_dec(data(x), n, a...)

Base.float(x::TrackedReal) = x

Base.promote_rule(::Type{TrackedReal{S}},::Type{T}) where {S,T} =
  TrackedReal{promote_type(S,T)}

using Random

Random.rand(x::Flux.Tracker.TrackedReal} = rand(typeof(x))
Random.rand(::Type{Flux.Tracker.TrackedReal{T}}) where {T} = convert(Flux.Tracker.TrackedReal{T},rand(T))
Random.rand(rng::AbstractRNG,x::Flux.Tracker.TrackedReal} = rand(rng,typeof(x))
Random.rand(rng::AbstractRNG,::Type{Flux.Tracker.TrackedReal{T}}) where {T} = convert(Flux.Tracker.TrackedReal{T},rand(rng,T))

Random.randn(x::Flux.Tracker.TrackedReal} = randn(typeof(x))
Random.randn(::Type{Flux.Tracker.TrackedReal{T}}) where {T} = convert(Flux.Tracker.TrackedReal{T},randn(T))
Random.randn(rng::AbstractRNG,x::Flux.Tracker.TrackedReal} = randn(rng,typeof(x))
Random.randn(rng::AbstractRNG,::Type{Flux.Tracker.TrackedReal{T}}) where {T} = convert(Flux.Tracker.TrackedReal{T},randn(rng,T))

Random.randexp(x::Flux.Tracker.TrackedReal} = randexp(typeof(x))
Random.randexp(::Type{Flux.Tracker.TrackedReal{T}}) where {T} = convert(Flux.Tracker.TrackedReal{T},randexp(T))
Random.randexp(rng::AbstractRNG,x::Flux.Tracker.TrackedReal} = randexp(rng,typeof(x))
Random.randexp(rng::AbstractRNG,::Type{Flux.Tracker.TrackedReal{T}}) where {T} = convert(Flux.Tracker.TrackedReal{T},randexp(rng,T))

using DiffRules, SpecialFunctions, NaNMath

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  @eval begin
    @grad $M.$f(a::Real) =
      $M.$f(data(a)), Δ -> (Δ * $(DiffRules.diffrule(M, f, :a)),)
    $M.$f(a::TrackedReal) = track($M.$f, a)
  end
end

# Work around zero(π) not working, for some reason
_zero(::Irrational) = nothing
_zero(x) = zero(x)

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :a, :b)
  f = :($M.$f)
  @eval begin
    @grad $f(a::TrackedReal, b::TrackedReal) = $f(data(a), data(b)), Δ -> (Δ * $da, Δ * $db)
    @grad $f(a::TrackedReal, b::Real) = $f(data(a), b), Δ -> (Δ * $da, _zero(b))
    @grad $f(a::Real, b::TrackedReal) = $f(a, data(b)), Δ -> (_zero(a), Δ * $db)
    $f(a::TrackedReal, b::TrackedReal)  = track($f, a, b)
    $f(a::TrackedReal, b::Real) = track($f, a, b)
    $f(a::Real, b::TrackedReal) = track($f, a, b)
  end
end

# Eliminating ambiguity
import Base:^

^(a::TrackedReal, b::Integer) = track(^, a, b)

# Tuples

struct TrackedTuple{T<:Tuple}
  data::T
  tracker::Tracked{T}
end

data(xs::TrackedTuple) = xs.data
tracker(xs::TrackedTuple) = xs.tracker

accum!(x::Tuple, Δ::Tuple) = accum!.(x, Δ)
init_grad(x::Tuple) = init_grad.(x)
zero_grad!(x::Tuple) = zero_grad!.(x)

track(f::Call, xs::Tuple) = TrackedTuple(xs, Tracked{typeof(xs)}(f, zero.(xs)))

function Base.show(io::IO, xs::TrackedTuple)
  show(io, data(xs))
  print(io, " (tracked)")
end

Base.length(x::TrackedTuple) = length(data(x))

Base.getindex(xs::TrackedTuple, i::Integer) = track(getindex, xs, i)

@grad function getindex(xs::TrackedTuple, i)
  data(xs)[i], Δ -> (ntuple(j -> i == j ? Δ : 0, length(xs)), nothing)
end

# Array collection

function collect(xs)
  xs = Base.collect(xs)
  track(Call(collect, (tracker.(xs),)), data.(xs))
end

function scan(c::Call{typeof(collect)})
  foreach(scan, c.args[1])
end

function back_(c::Call{typeof(collect)}, Δ, once)
  foreach((x, d) -> back(x, d, once), c.args[1], data(Δ))
end

function back_(g::Grads, c::Call{typeof(collect)}, Δ)
  foreach((x, Δ) -> back(g, x, Δ), c.args[1], Δ)
end
