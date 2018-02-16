struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  tracker::Tracked{A}
  data::A
  grad::A
  TrackedArray{T,N,A}(t::Tracked{A}, data::A) where {T,N,A} = new(t, data)
  TrackedArray{T,N,A}(t::Tracked{A}, data::A, grad::A) where {T,N,A} = new(t, data, grad)
end

tracker(x::TrackedArray) = x.tracker

TrackedVector{T,A} = TrackedArray{T,1,A}
TrackedMatrix{T,A} = TrackedArray{T,2,A}
TrackedVecOrMat{T,A} = Union{TrackedVector{T,A},TrackedMatrix{T,A}}

track(c::Call, x::AbstractArray) = TrackedArray(c, x)

TrackedArray(c::Call, x::A) where A <: AbstractArray =
  TrackedArray{eltype(A),ndims(A),A}(Tracked{A}(c, x), x)

TrackedArray(c::Call, x::A, Δ::A) where A <: AbstractArray =
  TrackedArray{eltype(A),ndims(A),A}(Tracked{A}(c, x, Δ), x, Δ)

TrackedArray(x::AbstractArray) = TrackedArray(Call(nothing), x, zeros(x))

Base.eltype(x::Type{<:TrackedArray{T}}) where T <: Real = TrackedReal{T}

Base.show(io::IO, ::Type{TrackedArray{T,N,A}}) where {T,N,A<:AbstractArray{T,N}} =
  print(io, "TrackedArray{…,$A}")

function Base.showarray(io::IO, X::TrackedArray, repr::Bool = true; header = true)
  if repr
    print(io, "param(")
    Base.showarray(io, data(X), true)
    print(io, ")")
  else
    header && print(io, "Tracked ")
    Base.showarray(io, data(X), false, header = header)
  end
end

Base.setindex!(xs::TrackedArray, v, i...) =
  error("Can't differentiate `setindex!`")

back!(::TrackedArray) = error("Use back!(x, Δ)")

# Fallthrough methods

for f in :[Base.size, Base.ndims].args
  @eval @inline $f(x::TrackedArray, a...) = $f(data(x), a...)
end

Base.similar(x::TrackedArray, dims::Union{AbstractUnitRange,Integer}...) =
  similar(data(x), dims...)

Base.similar(x::TrackedArray, T::Type) = similar(data(x), T)

Base.:(==)(x::TrackedArray, y) = data(x) == y
Base.:(==)(y, x::TrackedArray) = y == data(x)
Base.:(==)(x::TrackedArray, y::TrackedArray) = data(x) == data(y)

# Array Stdlib

Base.getindex(xs::TrackedArray, i...) = track(getindex, xs, i...)

function back(::typeof(getindex), Δ, xs::TrackedArray, i...)
  Δ′ = zeros(xs.data)
  Δ′[i...] = Δ
  @back(xs, Δ′)
end

Base.:-(xs::TrackedArray) = track(-, xs)

back(::typeof(-), Δ, xs::TrackedArray) = back(xs, -Δ)

Base.transpose(xs::TrackedArray) = track(transpose, xs)
Base.ctranspose(xs::TrackedArray) = track(ctranspose, xs)

back(::typeof(transpose), Δ, xs) = @back(xs, trim(xs, Δ.'))
back(::typeof(ctranspose), Δ, xs) = @back(xs, trim(xs, Δ'))

Base.repmat(x::TrackedVecOrMat, a::Integer...) = track(repmat, x, a...)
Base.repmat(x::TrackedVecOrMat, a::Int64...) = track(repmat, x, a...)

Base.vcat(a::TrackedVector, b::TrackedVector)  = track(vcat, a, b)
Base.vcat(a::TrackedVector, b::TrackedVector...)  = track(vcat, a, b...)
Base.vcat(a::TrackedVector, b::AbstractVector) = track(vcat, a, b)
Base.vcat(a::AbstractVector, b::TrackedVector) = track(vcat, a, b)

Base.vcat(a::TrackedVecOrMat, b::TrackedVecOrMat)  = track(vcat, a, b)
Base.vcat(a::TrackedVecOrMat, b::TrackedVecOrMat...)  = track(vcat, a, b...)
Base.vcat(a::TrackedVecOrMat, b::AbstractVecOrMat) = track(vcat, a, b)
Base.vcat(a::AbstractVecOrMat, b::TrackedVecOrMat) = track(vcat, a, b)

Base.vcat(a::TrackedMatrix, b::TrackedMatrix)  = track(vcat, a, b)
Base.vcat(a::TrackedMatrix, b::TrackedMatrix...)  = track(vcat, a, b...)
Base.vcat(a::TrackedMatrix, b::AbstractMatrix) = track(vcat, a, b)
Base.vcat(a::AbstractMatrix, b::TrackedMatrix) = track(vcat, a, b)

function back(::typeof(vcat), Δ, xs...)
  i = Base.tail(map(_ -> :, size(Δ)))
  start = 0
  for xsi in xs
    @back(xsi, Δ[start+1:start+size(xsi,1), i...])
    start += size(xsi, 1)
  end
end

Base.reshape(xs::TrackedArray, dims::Union{Colon,Int64}...) =
  track(reshape, xs, dims...)

Base.reshape(xs::TrackedArray, dims::Tuple{Vararg{Int64,N}} where N) =
  track(reshape, xs, dims)

back(::typeof(reshape), Δ, xs::TrackedArray, _...) =
  back(xs, reshape(Δ, size(xs)))


function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end

Base.kron(a::TrackedMatrix, b::TrackedMatrix)  = _kron(a, b)
Base.kron(a::TrackedMatrix, b::AbstractMatrix) = _kron(a, b)
Base.kron(a::AbstractMatrix, b::TrackedMatrix) = _kron(a, b)

# Reductions

Base.sum(xs::TrackedArray, dim) = track(sum, xs, dim)
Base.sum(xs::TrackedArray) = track(sum, xs)
Base.sum(f::Union{Function,Type},xs::TrackedArray) = sum(f.(xs))

back(::typeof(sum), Δ, xs::TrackedArray, dim...) = back(xs, similar(xs.data) .= Δ)

Base.maximum(xs::TrackedArray, args...) = maximum(xs.data, args...)
Base.findfirst(xs::TrackedArray, args...) = findfirst(xs.data, args...)

Base.mean(xs::TrackedArray) = track(mean, xs)
Base.mean(xs::TrackedArray, region) = track(mean, xs, region)

LinAlg.dot(xs::TrackedVector, ys::TrackedVector) = track(dot, xs, ys)
LinAlg.dot(xs::AbstractVector, ys::TrackedVector) = track(dot, xs, ys)
LinAlg.dot(xs::TrackedVector, ys::AbstractVector) = track(dot, xs, ys)

function back(::typeof(dot), Δ, xs, ys)
  @back(xs, Δ.*data(ys))
  @back(ys, Δ.*data(xs))
end

# Hacks to get std working
Base.std(x::TrackedArray; mean = Base.mean(x)) =
  sqrt.(sum((x .- mean).^2) ./ (length(x)-1))
Base.std(x::TrackedArray, dim; mean = Base.mean(x, dim)) =
  sqrt.(sum((x .- mean).^2, dim) ./ (size(x, dim)-1))

Base.norm(x::TrackedArray, p::Real = 2) =
  p == 1 ? sum(abs.(x)) :
  p == 2 ? sqrt(sum(abs2.(x))) :
  error("$p-norm not supported")

back(::typeof(mean), Δ, xs::TrackedArray) = back(xs, similar(xs.data) .= Δ ./ length(xs.data))
back(::typeof(mean), Δ, xs::TrackedArray, region) =
  back(xs, similar(xs.data) .= Δ ./ prod(size(xs.data, region...)))

# BLAS

Base.diagm(x::TrackedVector) = track(diagm, x)
back(::typeof(diagm), Δ, x) = @back(x, diag(Δ))

for f in :[*, Ac_mul_B, A_mul_Bc].args
  @eval begin
    import Base.$f
    $f(a::TrackedMatrix, b::TrackedMatrix)  = track($f, a, b)
    $f(a::TrackedMatrix, b::AbstractMatrix) = track($f, a, b)
    $f(a::AbstractMatrix, b::TrackedMatrix) = track($f, a, b)

    $f(a::TrackedMatrix, b::TrackedVector)  = track($f, a, b)
    $f(a::TrackedMatrix, b::AbstractVector) = track($f, a, b)
    $f(a::AbstractMatrix, b::TrackedVector) = track($f, a, b)

    $f(a::TrackedVector, b::TrackedVector)  = track($f, a, b)
    $f(a::TrackedVector, b::AbstractVector) = track($f, a, b)
    $f(a::AbstractVector, b::TrackedVector) = track($f, a, b)
  end
end

function back(::typeof(*), Δ, a::AbstractMatrix, b::AbstractVecOrMat)
  @back(a, A_mul_Bt(Δ, data(b)))
  @back(b, At_mul_B(data(a), Δ))
end

function back(::typeof(Ac_mul_B), Δ, a::AbstractVecOrMat{<:Real}, b::AbstractVecOrMat{<:Real})
  @back(a, A_mul_Bt(Δ, data(b))')
  @back(b, data(a)*Δ)
end

function back(::typeof(A_mul_Bc), Δ, a::AbstractVecOrMat{<:Real}, b::AbstractVecOrMat{<:Real})
  @back(a, Δ * data(b))
  @back(b, At_mul_B(data(a), Δ)')
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

using NNlib
import NNlib: softmax, ∇softmax, logsoftmax, ∇logsoftmax, conv2d, pool

softmax(xs::TrackedArray) = track(softmax, xs)

back(::typeof(softmax), Δ, xs) = @back(xs, ∇softmax(Δ, data(xs)))

logsoftmax(xs::TrackedArray) = track(logsoftmax, xs)

back(::typeof(logsoftmax), Δ, xs) = @back(xs, ∇logsoftmax(Δ, data(xs)))

# TODO: can store kwargs efficiently in namedtuples
_conv2d(x, w, stride, pad) = conv2d(x, w, stride = stride, padding = pad)

conv2d(x::TrackedArray{<:Any,4}, w::TrackedArray{<:Any,4}; stride = 1, padding = 0) =
  track(_conv2d, x, w, stride, padding)
conv2d(x::AbstractArray{<:Any,4}, w::TrackedArray{<:Any,4}; stride = 1, padding = 0) =
  track(_conv2d, x, w, stride, padding)
conv2d(x::TrackedArray{<:Any,4}, w::AbstractArray{<:Any,4}; stride = 1, padding = 0) =
  track(_conv2d, x, w, stride, padding)

function back(::typeof(_conv2d), Δ, x, w, stride, pad)
  @back(x, NNlib.conv2d_grad_x(data(x), data(w), Δ; stride = stride, padding = pad))
  @back(w, NNlib.conv2d_grad_w(data(x), data(w), Δ; stride = stride, padding = pad))
end

_pool(x, k, pad, mode) = pool(x, window = k, mode = mode, padding = pad)

pool(x::TrackedArray{<:Any,4}; window = 2, mode = 0, padding = 0) =
  track(_pool, x, window, padding, mode)

back_(::typeof(_pool), y, Δ, x, k, pad, mode) =
  back(x, NNlib.pool_grad(data(x), y, Δ, window=k, mode=mode, padding=pad))

# Broadcasting

using ForwardDiff: Dual, partials

struct Broadcasted{F,T}
  f::F
  data::T
end

(b::Broadcasted)(xs...) = map(x -> x.value, b.data)

dualify(xs, n) = xs
dualify(xs::TrackedArray, ps) = map(x -> Dual(x, ps), data(xs))
dualify(xs::TrackedReal, ps) = Dual(data(xs), ps)

function tracked_broadcast(f, args::Vararg{Any,N}) where N
  dargs = map((x,i) -> dualify(x, ntuple(j -> i==j, Val{N})), args, ntuple(identity, Val{N}))
  out = broadcast(f, dargs...)
  eltype(out) <: Dual || return out
  b = Broadcasted(f, out)
  track(Call(b, args...), b())
end

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val{ndims(x)}))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
    trim(x, sum(Δ, filter(n -> size(x, n) == 1, 1:ndims(Δ))))

unbroadcast(x::Number, Δ) = sum(Δ)

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
