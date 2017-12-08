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
Base.vcat(a::TrackedVector, b::TrackedVector...)  = TrackedArray(Call(vcat, a, b...))
Base.vcat(a::TrackedVector, b::AbstractVector) = TrackedArray(Call(vcat, a, b))
Base.vcat(a::AbstractVector, b::TrackedVector) = TrackedArray(Call(vcat, a, b))

Base.vcat(a::TrackedVecOrMat, b::TrackedVecOrMat)  = TrackedArray(Call(vcat, a, b))
Base.vcat(a::TrackedVecOrMat, b::AbstractVecOrMat) = TrackedArray(Call(vcat, a, b))
Base.vcat(a::AbstractVecOrMat, b::TrackedVecOrMat) = TrackedArray(Call(vcat, a, b))

Base.vcat(a::TrackedMatrix, b::TrackedMatrix)  = TrackedArray(Call(vcat, a, b))
Base.vcat(a::TrackedMatrix, b::AbstractMatrix) = TrackedArray(Call(vcat, a, b))
Base.vcat(a::AbstractMatrix, b::TrackedMatrix) = TrackedArray(Call(vcat, a, b))

function back(::typeof(vcat), Δ, xs...)
  i = Base.tail(map(_ -> :, size(Δ)))
  start = 0
  for xsi in xs
    @back(xsi, Δ[start+1:start+size(xsi,1), i...])
    start += size(xsi, 1)
  end
end

# Reductions

Base.sum(xs::TrackedArray, dim) = TrackedArray(Call(sum, xs, dim))
Base.sum(xs::TrackedArray) = TrackedArray(Call(sum, xs), toarray(xs.data, sum(xs.data)))
Base.sum(xs::TrackedScalar, dim...) = xs

back(::typeof(sum), Δ, xs::TrackedArray, dim...) = back(xs, similar(xs.data) .= Δ)

Base.maximum(xs::TrackedArray, args...) = maximum(xs.data, args...)
Base.findfirst(xs::TrackedArray, args...) = findfirst(xs.data, args...)

Base.mean(xs::TrackedArray) = TrackedArray(Call(mean, xs), toarray(xs.data, mean(xs.data)))
Base.mean(xs::TrackedArray, region) = TrackedArray(Call(mean, xs, region))

# Hacks to get std working
Base.std(x::TrackedArray; mean = Base.mean(x)) =
  sqrt.(sum((x .- mean).^2) ./ (length(x)-1))
Base.std(x::TrackedArray, dim; mean = Base.mean(x, dim)) =
  sqrt.(sum((x .- mean).^2, dim) ./ (size(x, dim)-1))

back(::typeof(mean), Δ, xs::TrackedArray) = back(xs, similar(xs.data) .= Δ ./ length(xs.data))
back(::typeof(mean), Δ, xs::TrackedArray, region) =
  back(xs, similar(xs.data) .= Δ ./ prod(size(xs.data, region...)))

# BLAS

for f in :[*, Ac_mul_B].args
  @eval begin
    import Base.$f
    $f(a::TrackedMatrix, b::TrackedMatrix)  = TrackedArray(Call($f, a, b))
    $f(a::TrackedMatrix, b::AbstractMatrix) = TrackedArray(Call($f, a, b))
    $f(a::AbstractMatrix, b::TrackedMatrix) = TrackedArray(Call($f, a, b))

    $f(a::TrackedMatrix, b::TrackedVector)  = TrackedArray(Call($f, a, b))
    $f(a::TrackedMatrix, b::AbstractVector) = TrackedArray(Call($f, a, b))
    $f(a::AbstractMatrix, b::TrackedVector) = TrackedArray(Call($f, a, b))

    $f(a::TrackedVector, b::TrackedVector)  = TrackedArray(Call($f, a, b))
    $f(a::TrackedVector, b::AbstractVector) = TrackedArray(Call($f, a, b))
    $f(a::AbstractVector, b::TrackedVector) = TrackedArray(Call($f, a, b))
  end
end

function back(::typeof(*), Δ, a::AbstractMatrix, b::AbstractVecOrMat)
  @back(a, A_mul_Bt(Δ, data(b)))
  @back(b, At_mul_B(data(a), Δ))
end

function back(::typeof(Ac_mul_B), Δ, a::AbstractVecOrMat{<:Real}, b::AbstractVecOrMat{<:Real})
  @back(a, A_mul_Bt(Δ, data(b))')
  @back(b, *(data(a), Δ))
end

# Fast path for matrix-vector
function back(::typeof(*), Δ::AbstractVector, W::TrackedMatrix, x::AbstractVector)
  if isleaf(W)
    W.grad .+= Δ .* data(x).'
  else
    back(W, A_mul_Bt(Δ, data(x)))
  end
  @back(x, At_mul_B(data(W), Δ))
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
