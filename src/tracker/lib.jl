import Base: *

toarray(xs::AbstractArray, ys::AbstractArray) = ys
toarray(xs::AbstractArray, y) = similar(xs, typeof(y), ()) .= y

unarray(xs) = xs
unarray(xs::AbstractArray{T,0} where T) = xs[]

Base.getindex(xs::TrackedArray, i...) =
  TrackedArray(Call(getindex, xs, i...), toarray(xs.data, xs.data[i...]))

function back(::typeof(getindex), Δ, xs::TrackedArray, i...)
  Δ′ = zeros(xs.data)
  Δ′[i...] = unarray(Δ)
  @back(xs, Δ′)
end

Base.:-(xs::TrackedArray) = TrackedArray(Call(-, xs))

back(::typeof(-), Δ, xs::TrackedArray) = back(xs, -Δ)

Base.transpose(xs::TrackedArray) = TrackedArray(Call(transpose, xs))
Base.ctranspose(xs::TrackedArray) = TrackedArray(Call(ctranspose, xs))

back(::typeof(transpose), Δ, xs) = @back(xs, trim(xs, Δ.'))
back(::typeof(ctranspose), Δ, xs) = @back(xs, trim(xs, Δ'))

Base.repmat(x::TrackedVecOrMat, a::Integer...) = TrackedArray(Call(repmat, x, a...))
Base.repmat(x::TrackedVecOrMat, a::Int64...) = TrackedArray(Call(repmat, x, a...))

Base.vcat(a::TrackedVector, b::TrackedVector)  = TrackedArray(Call(vcat, a, b))
Base.vcat(a::TrackedVector...)                 = TrackedArray(Call(vcat, a...))
Base.vcat(a::TrackedVector, b::AbstractVector) = TrackedArray(Call(vcat, a, b))
Base.vcat(a::AbstractVector, b::TrackedVector) = TrackedArray(Call(vcat, a, b))

Base.vcat(a::TrackedVecOrMat, b::TrackedVecOrMat)  = TrackedArray(Call(vcat, a, b))
Base.vcat(a::TrackedVecOrMat, b::AbstractVecOrMat) = TrackedArray(Call(vcat, a, b))
Base.vcat(a::AbstractVecOrMat, b::TrackedVecOrMat) = TrackedArray(Call(vcat, a, b))

Base.vcat(a::TrackedMatrix, b::TrackedMatrix)  = TrackedArray(Call(vcat, a, b))
Base.vcat(a::TrackedMatrix, b::AbstractMatrix) = TrackedArray(Call(vcat, a, b))
Base.vcat(a::AbstractMatrix, b::TrackedMatrix) = TrackedArray(Call(vcat, a, b))

function back(::typeof(vcat), Δ, xs, ys)
  i = Base.tail(map(_ -> :, size(Δ)))
  @back(xs, Δ[1:size(xs,1), i...])
  @back(ys, Δ[size(xs,1)+1:end, i...])
end

# Reductions

Base.sum(xs::TrackedArray, dim) = TrackedArray(Call(sum, xs, dim))
Base.sum(xs::TrackedArray) = TrackedArray(Call(sum, xs), toarray(xs.data, sum(xs.data)))
Base.sum(xs::TrackedScalar, dim...) = xs

back(::typeof(sum), Δ, xs::TrackedArray, dim...) = back(xs, similar(xs.data) .= Δ)

Base.maximum(xs::TrackedArray, args...) = maximum(xs.data, args...)
Base.findfirst(xs::TrackedArray, args...) = findfirst(xs.data, args...)

# BLAS

a::TrackedMatrix * b::TrackedMatrix  = TrackedArray(Call(*, a, b))
a::TrackedMatrix * b::AbstractMatrix = TrackedArray(Call(*, a, b))
a::AbstractMatrix * b::TrackedMatrix = TrackedArray(Call(*, a, b))

a::TrackedMatrix * b::TrackedVector  = TrackedArray(Call(*, a, b))
a::TrackedMatrix * b::AbstractVector = TrackedArray(Call(*, a, b))
a::AbstractMatrix * b::TrackedVector = TrackedArray(Call(*, a, b))

function back(::typeof(*), Δ, a::AbstractMatrix, b::AbstractVecOrMat)
  @back(a, A_mul_Bt(Δ, data(b)))
  @back(b, At_mul_B(data(a), Δ))
end

# NNlib

import NNlib: softmax, ∇softmax

softmax(xs::TrackedArray) = TrackedArray(Call(softmax, xs))

back(::typeof(softmax), Δ, xs) = @back(xs, ∇softmax(Δ, data(xs)))

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

function back(b::Broadcasted, Δ, args::Vararg{Any,N}) where N
  Δargs = ntuple(i -> getpartial.(Δ, b.data, i), Val{N})
  foreach((x, Δ) -> @back(x, unbroadcast(x, Δ)), args, Δargs)
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
