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

back!(::typeof(-), Δ, xs::TrackedArray) = back!(xs, -Δ)

Base.transpose(xs::TrackedArray) = TrackedArray(Call(transpose, xs))
Base.ctranspose(xs::TrackedArray) = TrackedArray(Call(ctranspose, xs))

# Reductions

Base.sum(xs::TrackedArray, dim) = TrackedArray(Call(sum, xs, dim))
Base.sum(xs::TrackedArray) = TrackedArray(Call(sum, xs), toarray(xs.x, sum(xs.x)))
Base.sum(xs::TrackedScalar, dim...) = xs

back!(::typeof(sum), Δ, xs::TrackedArray, dim...) = back!(xs, similar(xs.x) .= Δ)

Base.maximum(xs::TrackedArray, args...) = maximum(xs.x, args...)
Base.findfirst(xs::TrackedArray, args...) = findfirst(xs.x, args...)

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

# concat

tryfind(e, c, ft, ff) = (i = findfirst(c, e); i != 0 ? ft(i) : ff(i); i)

function back!(::typeof(cat), Δ, dims, xs...)
  acc = map(zero, dims)
  for (i, x) in enumerate(xs)
    slice = map(d->size(x, d), dims)
    @back!(x, view(Δ, [tryfind(d, dims, i->1+acc[i]:slice[i], i->:) for d in 1:ndims(x)]...))
    acc = acc .+ slice
  end
end

# NNlib

import NNlib: softmax, ∇softmax

softmax(xs::TrackedArray) = TrackedArray(Call(softmax, xs))

back!(::typeof(softmax), Δ, xs) = @back!(xs, ∇softmax(Δ, data(xs)))

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

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val{ndims(x)}))

unbroadcast(x, Δ) =
  size(x) == size(Δ) ? Δ :
    trim(x, sum(Δ, filter(n -> size(x, n) == 1, 1:ndims(Δ))))

function getpartial(Δ, x, i)
  @inbounds p = getindex(partials(x), i)
  return Δ * p
end

function back!(b::Broadcasted, Δ, args::Vararg{Any,N}) where N
  Δargs = ntuple(i -> getpartial.(Δ, b.data, i), Val{N})
  foreach((x, Δ) -> @back!(x, unbroadcast(x, Δ)), args, Δargs)
end

Base.Broadcast._containertype(::Type{<:TrackedArray}) = TrackedArray
Base.Broadcast.promote_containertype(::Type{TrackedArray}, ::Type{TrackedArray}) = TrackedArray
Base.Broadcast.promote_containertype(::Type{Array}, ::Type{TrackedArray}) = TrackedArray
Base.Broadcast.promote_containertype(::Type{TrackedArray}, ::Type{Array}) = TrackedArray
Base.Broadcast.promote_containertype(::Type{TrackedArray}, ct) = TrackedArray
Base.Broadcast.promote_containertype(ct, ::Type{TrackedArray}) = TrackedArray
Base.Broadcast.broadcast_indices(::Type{TrackedArray}, A::Ref) = ()
Base.Broadcast.broadcast_indices(::Type{TrackedArray}, A) = indices(A)

Base.Broadcast.broadcast_c(f, ::Type{TrackedArray}, A, Bs...) = tracked_broadcast(f, A, Bs...)
