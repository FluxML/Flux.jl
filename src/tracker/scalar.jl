struct TrackedReal{T<:Real} <: Real
  tracker::Tracked{T}
end

TrackedReal(x::Real) = TrackedReal(Tracked(Call(nothing), x, zero(x)))

tracker(x::TrackedReal) = x.tracker

track(f::Call, x::Real) = TrackedReal(Tracked(f, x, zero(x)))

back!(x::TrackedReal) = back!(x, 1)

function Base.show(io::IO, x::TrackedReal)
  show(io, data(x))
  print(io, " (tracked)")
end

Base.decompose(x::TrackedReal) = Base.decompose(data(x))

Base.convert(::Type{TrackedReal{T}}, x::TrackedReal{T}) where T = x

Base.convert(::Type{TrackedReal{T}}, x::TrackedReal) where T =
  TrackedReal(Tracked(x.tracker.f, convert(T, x.tracker.data)))

Base.convert(::Type{TrackedReal{T}}, x::Real) where T = TrackedReal(convert(T, x))

Base.:(<)(x::TrackedReal, y::TrackedReal) = data(x) < data(y)
Base.:(==)(x::TrackedReal, y::TrackedReal) = data(x) == data(y)

for f in :[isinf, isnan, isfinite].args
  @eval Base.$f(x::TrackedReal) = Base.$f(data(x))
end

Base.Printf.fix_dec(x::TrackedReal, n::Int) = Base.Printf.fix_dec(data(x), n)

Base.promote_rule(::Type{TrackedReal{S}},::Type{T}) where {S,T} =
  TrackedReal{promote_type(S,T)}

using DiffRules, SpecialFunctions, NaNMath

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  @eval begin
    $M.$f(a::TrackedReal) = track($M.$f, a)
    back(::typeof($M.$f), Δ::Real, a::TrackedReal) =
      back(a, Δ * $(DiffRules.diffrule(M, f, :(data(a)))))
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :(data(a)), :(data(b)))
  @eval begin
    $M.$f(a::TrackedReal, b::TrackedReal)  = track($M.$f, a, b)
    $M.$f(a::TrackedReal, b::Real) = track($M.$f, a, b)
    $M.$f(a::Real, b::TrackedReal) = track($M.$f, a, b)
    function back(::typeof($M.$f), Δ::Real, a::Real, b::Real)
      @back(a, Δ * $da)
      @back(b, Δ * $db)
    end
  end
end

# Tuples

struct TrackedTuple{T<:Tuple}
  tracker::Tracked{T}
end

tracker(xs::TrackedTuple) = xs.tracker

accum!(x::Tuple, Δ::Tuple) = accum!.(x, Δ)
init_grad(x::Tuple) = init_grad.(x)
zero_grad!(x::Tuple) = zero_grad!.(x)

track(f::Call, xs::Tuple) = TrackedTuple(Tracked(f, xs))

function Base.show(io::IO, xs::TrackedTuple)
  show(io, data(xs))
  print(io, " (tracked)")
end

Base.length(x::TrackedTuple) = length(data(x))

Base.getindex(xs::TrackedTuple, i::Integer) = track(getindex, xs, i)

back(::typeof(getindex), Δ, t, i) =
  back(t, ntuple(j -> i == j ? Δ : 0, length(t)))
