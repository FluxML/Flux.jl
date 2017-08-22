import Base: *

toarray(xs::AbstractArray, ys::AbstractArray) = ys
toarray(xs::AbstractArray, y) = similar(xs, typeof(y), ()) .= y

Base.getindex(xs::TrackedArray, i...) =
  TrackedArray(Call(getindex, xs, i...), toarray(xs.x, xs.x[i...]))

function back!(::typeof(getindex), Δ, xs::TrackedArray, i...)
  Δ′ = zeros(xs.x)
  Δ′[i...] = Δ
  @back!(xs, Δ′)
end

Base.:-(xs::TrackedArray) = TrackedArray(Call(-, xs))

# Reductions

Base.sum(xs::TrackedArray, dim) = TrackedArray(Call(sum, xs, dim))
Base.sum(xs::TrackedArray) = TrackedArray(Call(sum, xs), toarray(xs.x, sum(xs.x)))

back!(::typeof(sum), Δ, xs::TrackedArray, dim...) = back!(xs, similar(xs.x) .= Δ)

# BLAS

a::TrackedMatrix * b::TrackedMatrix  = TrackedArray(Call(*, a, b))
a::TrackedMatrix * b::AbstractMatrix = TrackedArray(Call(*, a, b))
a::AbstractMatrix * b::TrackedMatrix = TrackedArray(Call(*, a, b))

a::TrackedMatrix * b::TrackedVector  = TrackedArray(Call(*, a, b))
a::TrackedMatrix * b::AbstractVector = TrackedArray(Call(*, a, b))
a::AbstractMatrix * b::TrackedVector = TrackedArray(Call(*, a, b))

function back!(::typeof(*), Δ, a::AbstractMatrix, b::AbstractVecOrMat)
  @back!(a, A_mul_Bt(Δ, data(b)))
  @back!(b, At_mul_B(data(a), Δ))
end

# Broadcasting

using ForwardDiff: Dual, partials

struct Broadcasted{T}
  data::T
end

(b::Broadcasted)(xs...) = map(x -> x.value, b.data)

dualify(xs, n) = xs
dualify(xs::TrackedArray, ps) = map(x -> Dual(x, ps), data(xs))

function tracked_broadcast(f, args::Vararg{Any,N}) where N
  dargs = map((x,i) -> dualify(x, ntuple(j -> i==j, Val{N})), args, ntuple(identity, Val{N}))
  # TrackedArray(Call(Broadcasted(broadcast(f, dargs...)), args...))
  # Works around a 0.6 type inference issue
  b = Broadcasted(broadcast(f, dargs...))
  TrackedArray(Call(b, args...), b())
end

function back!(b::Broadcasted, Δ, args...)
  Δargs = ntuple(i -> Δ .* getindex.(partials.(b.data), i), length(args))
  map((x, Δ) -> @back!(x, Δ), args, Δargs)
  return
end

Base.Broadcast._containertype(::Type{<:TrackedArray}) = TrackedArray
Base.Broadcast.promote_containertype(::Type{Array}, ::Type{TrackedArray}) = TrackedArray
Base.Broadcast.promote_containertype(::Type{TrackedArray}, ::Type{Array}) = TrackedArray
Base.Broadcast.broadcast_indices(::Type{TrackedArray}, A::Ref) = ()
Base.Broadcast.broadcast_indices(::Type{TrackedArray}, A) = indices(A)

Base.Broadcast.broadcast_c(f, ::Type{TrackedArray}, A, Bs...) = tracked_broadcast(f, A, Bs...)
