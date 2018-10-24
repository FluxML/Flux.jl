import Base: *

import LinearAlgebra
import LinearAlgebra: inv, \, /

using Statistics
using LinearAlgebra: Transpose, Adjoint, diagm, diag

struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  tracker::Tracked{A}
  data::A
  grad::A
  TrackedArray{T,N,A}(t::Tracked{A}, data::A) where {T,N,A} = new(t, data)
  TrackedArray{T,N,A}(t::Tracked{A}, data::A, grad::A) where {T,N,A} = new(t, data, grad)
end

data(x::TrackedArray) = x.data
tracker(x::TrackedArray) = x.tracker

TrackedVector{T,A} = TrackedArray{T,1,A}
TrackedMatrix{T,A} = TrackedArray{T,2,A}
TrackedVecOrMat{T,A} = Union{TrackedVector{T,A},TrackedMatrix{T,A}}

track(c::Call, x::AbstractArray) = TrackedArray(c, x)

TrackedArray(c::Call, x::A) where A <: AbstractArray =
  TrackedArray{eltype(A),ndims(A),A}(Tracked{A}(c), x)

TrackedArray(c::Call, x::A, Δ::A) where A <: AbstractArray =
  TrackedArray{eltype(A),ndims(A),A}(Tracked{A}(c, Δ), x, Δ)

TrackedArray(x::AbstractArray) = TrackedArray(Call(), x, zero(x))

Base.eltype(x::Type{<:TrackedArray{T}}) where T <: Real = TrackedReal{T}

Base.show(io::IO, ::Type{TrackedArray{T,N,A}}) where {T,N,A<:AbstractArray{T,N}} =
  print(io, "TrackedArray{…,$A}")

function Base.summary(io::IO, x::TrackedArray)
  print(io, "Tracked ")
  summary(io, data(x))
end

Base.print_array(io::IO, x::TrackedArray) = Base.print_array(io, data(x))

Base.copy(x::TrackedArray) = x

Base.setindex!(xs::TrackedArray, v, i...) =
  error("Can't differentiate `setindex!`")

back!(::TrackedArray) = error("Value is not scalar; use `back!(sum(x))` or `back!(x, Δ)`")

# Fallthrough methods

for f in :[Base.size, Base.ndims, Base.collect].args
  @eval @inline $f(x::TrackedArray, a...) = $f(data(x), a...)
end

Base.size(x::TrackedArray, i::Integer, j::Integer, is::Integer...) =
  size(data(x), i, j, is...)

Base.similar(x::TrackedArray, dims::Union{AbstractUnitRange,Integer}...) =
  similar(data(x), dims...)

Base.similar(x::TrackedArray, T::Type) = similar(data(x), T)

for op in [:(==), :≈]
    @eval Base.$op(x::TrackedArray, y::AbstractArray) = Base.$op(data(x), y)
    @eval Base.$op(x::AbstractArray, y::TrackedArray) = Base.$op(x, data(y))
    @eval Base.$op(x::TrackedArray, y::TrackedArray) = Base.$op(data(x), data(y))
end

# Array Stdlib

Base.getindex(xs::TrackedArray, i...) = track(getindex, xs, i...)

@grad function getindex(xs::AbstractArray, i...)
  data(xs)[i...], function (Δ)
    Δ′ = zero(xs)
    Δ′[i...] = data(Δ)
    (nobacksies(:getindex, Δ′), map(_->nothing, i)...)
  end
end

Base.:-(xs::TrackedArray) = track(-, xs)

@grad -(xs) = -data(xs), Δ -> (-Δ,)

Base.transpose(xs::TrackedArray) = track(transpose, xs)
Base.adjoint(xs::TrackedArray) = track(adjoint, xs)

@grad transpose(xs) = transpose(data(xs)), Δ -> (reshape(transpose(Δ), size(xs)),)
@grad adjoint(xs) = data(xs)', Δ -> (reshape(Δ', size(xs)),)

Base.repeat(xs::TrackedArray; kw...) = track(repeat, xs; kw...)

@grad function repeat(xs; inner=ntuple(x->1, ndims(xs)), outer=ntuple(x->1, ndims(xs)))
  repeat(data(xs), inner = inner, outer = outer), function (Δ)
    Δ′ = zero(xs)
    S = size(xs)

    # Loop through each element of Δ, calculate source dimensions, accumulate into Δ′
    for (dest_idx, val) in pairs(IndexCartesian(), data(Δ))
        # First, round dest_idx[dim] to nearest gridpoint defined by inner[dim], then
        # wrap around based on original size S.
        src_idx = [mod1(div(dest_idx[dim] - 1, inner[dim]) + 1, S[dim]) for dim in 1:length(S)]
        Δ′[src_idx...] += val
    end
    (nobacksies(:repeat, Δ′),)
  end
end

for f in [:vcat, :hcat]
  UArray = :(Union{TrackedArray,Vector,Matrix,Adjoint,Transpose})
  @eval begin
    # This section is a bit of a hack since julia doesn't have a standardised
    # promotion mechanism for concatenation yet
    # https://github.com/JuliaLang/julia/pull/20815

    # It should support tracked concatenation with rank ∈ (1,2) with a
    # TrackedArray anywhere among the arguments This works as long as base has
    # other functions that captures `(::Union{Vector,RowVector,Matrix}...)`.
    Base.$f(a::$UArray...) = track($f, a...)

    # It should support tracked concatenation with rank>2 if the TrackedArray is
    # first
    Base.$f(a::TrackedArray, b::AbstractArray...) = track($f, a, b...)
    Base.$f(a::TrackedArray, b::$UArray...) = track($f, a, b...) # resolves ambiguity introduced by previous row

    # It should support tracked concatenation with rank>2 if the TrackedArray is
    # second
    Base.$f(a::Array, b::TrackedArray, c::AbstractArray...) = track($f, a, b, c...)
    Base.$f(a::Union{Vector,Matrix,Adjoint,Transpose}, b::TrackedArray,
            c::$UArray...) =
      track($f, a, b, c...) # resolves ambiguity introduced by previous row
  end
end

@grad function vcat(xs...)
  vcat(data.(xs)...), function (Δ)
    start = 0
    Δs = [begin
      i = map(_ -> :, size(xsi)) |> Base.tail
      d = Δ[start+1:start+size(xsi,1), i...]
      start += size(xsi, 1)
      d
    end for xsi in xs]
    return (Δs...,)
  end
end

@grad function hcat(xs...)
  hcat(data.(xs)...), function (Δ)
    start = 0
    Δs = [begin
      d = if ndims(xsi) == 1
        Δ[:, start+1]
      else
        i = map(_ -> :, size(xsi)) |> Base.tail |> Base.tail
        Δ[:, start+1:start+size(xsi,2), i...]
      end
      start += size(xsi, 2)
      d
    end for xsi in xs]
    return (Δs...,)
  end
end

Base.cat(a::TrackedArray; dims) = track(cat, a, dims = dims)
Base.cat(a::TrackedArray, b::TrackedArray, c::AbstractArray...; dims) = track(cat, a, b, c..., dims = dims)
Base.cat(a::TrackedArray, b::AbstractArray, c::AbstractArray...; dims) = track(cat, a, b, c..., dims = dims)
Base.cat(a::AbstractArray, b::TrackedArray, c::AbstractArray...; dims) = track(cat, a, b, c..., dims = dims)

@grad function cat(Xs...; dims)
  cat(data.(Xs)..., dims = dims), function (Δ)
    start = ntuple(i -> 0, Val(ndims(Δ)))
    Δs = [begin
      dim_xs = 1:ndims(xs)
      till_xs = ntuple((i -> i in dims ? (i in dim_xs ? size(xs,i) : 1) : 0), Val(ndims(Δ)))
      xs_in_Δ = ntuple(i -> till_xs[i] > 0 ? (start[i]+1:start[i]+till_xs[i]) : Colon(), Val(ndims(Δ)))
      d = reshape(Δ[xs_in_Δ...],size(xs))
      start = start .+ till_xs
      d
    end for xs in Xs]
    return (Δs...,)
  end
end

Base.reshape(xs::TrackedArray, dims::Union{Colon,Int64}...) = reshape(xs, dims)
Base.reshape(xs::TrackedArray, dims::Tuple{Vararg{Union{Int64,Colon}}}) = reshape(xs, Base._reshape_uncolon(xs, dims))
Base.reshape(xs::TrackedArray, dims::Tuple{Vararg{Int64}}) = track(reshape, xs, dims)

@grad reshape(xs, dims) = reshape(data(xs), dims), Δ -> (reshape(Δ, size(xs)),nothing)

Base.permutedims(xs::TrackedArray, dims) = track(permutedims, xs, dims)
@grad permutedims(xs, dims) = permutedims(data(xs), dims), Δ -> (permutedims(Δ, invperm(dims)),nothing)

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


inv(A::TrackedArray) = Tracker.track(inv, A)
@grad function inv(A)
    return inv(Tracker.data(A)), function (Δ)
        Ainv = inv(A)
        ∇A = - Ainv' * Δ * Ainv'
        return (∇A, )
    end
end

#       (/) rdivide
A::TrackedArray     / B::TrackedArray     = Tracker.track(/, A, B)
A::AbstractVecOrMat / B::TrackedArray     = Tracker.track(/, A, B)
A::TrackedArray     / B::AbstractVecOrMat = Tracker.track(/, A, B)
@grad function (A / B)
    return Tracker.data(A) / Tracker.data(B), function (Δ)
        Binv = inv(B)
        ∇B = - Binv' * A' * Δ * Binv'
        return (Δ * Binv',  ∇B)
    end
end

#       (\) ldivide  (left vec divide needs more work to resolve dispatch ambiguity)
A::TrackedArray     \ B::TrackedArray     = Tracker.track(\, A, B)
A::AbstractArray    \ B::TrackedArray     = Tracker.track(\, A, B)
A::TrackedArray     \ B::AbstractVecOrMat = Tracker.track(\, A, B)
@grad function (A \ B)
    return Tracker.data(A) \ Tracker.data(B), function (Δ)
        Ainv = inv(A)
        ∇A = - Ainv' * Δ * B' * Ainv'
        return (∇A,  Ainv' * Δ)
    end
end


# Reductions

Base.sum(xs::TrackedArray; dims = :) = track(sum, xs, dims = dims)
Base.sum(f::Union{Function,Type},xs::TrackedArray) = sum(f.(xs))

@grad sum(xs; dims = :) = sum(data(xs), dims = dims),
  Δ -> (zero(xs) .+ Δ, )

Base.prod(xs::TrackedArray, dim) = track(prod, xs, dim)
Base.prod(xs::TrackedArray) = track(prod, xs)
Base.prod(f::Union{Function, Type}, xs::TrackedArray) = prod(f.(xs))

@grad prod(xs) = prod(data(xs)), Δ -> (prod(xs) ./ xs .* Δ,)
@grad prod(xs, dim) = prod(data(xs), dims = dim),
  Δ -> (nobacksies(:sum,
          reshape(.*(circshift.([reshape(data(xs), length(xs))], 1:length(xs)-1)...), size(xs)) .* Δ),
        nothing)

Base.findfirst(xs::TrackedArray, args...) = findfirst(xs.data, args...)

Statistics.mean(xs::TrackedArray; dims = :) = track(mean, xs, dims = dims)

Base.maximum(xs::TrackedArray; dims = :) = track(maximum, xs, dims = dims)
Base.minimum(xs::TrackedArray; dims = :) = track(minimum, xs, dims = dims)

import LinearAlgebra: dot

dot(xs::TrackedVector, ys::TrackedVector) = track(dot, xs, ys)
dot(xs::AbstractVector, ys::TrackedVector) = track(dot, xs, ys)
dot(xs::TrackedVector, ys::AbstractVector) = track(dot, xs, ys)

@grad dot(xs, ys) = dot(data(xs), data(ys)), Δ -> (Δ .* ys, Δ .* xs)

# Hacks to get std working
Statistics.std(x::TrackedArray; dims = :, mean = Statistics.mean(x, dims = dims)) = _std(x,mean,dims)
_std(x::TrackedArray, mean, dims) = sqrt.(sum((x .- mean).^2, dims = dims) ./ (mapreduce(i -> size(x,i),*, dims) - 1))
_std(x::TrackedArray, mean, ::Colon) = sqrt.(sum((x .- mean).^2) ./ (length(x) - 1))

LinearAlgebra.norm(x::TrackedArray, p::Real = 2) =
  sum(abs.(x).^p .+ eps(0f0))^(1/p) # avoid d(sqrt(x))/dx == Inf at 0

@grad mean(xs; dims = :) = mean(data(xs), dims=dims), Δ -> (_backmean(xs,Δ,dims),)
_backmean(xs, Δ, ::Colon) = zero(xs) .+ Δ ./ length(xs)
_backmean(xs, Δ, dims) = zero(xs) .+ Δ ./ mapreduce(i -> size(data(xs),i),*,dims)

@grad function maximum(xs; dims = dims)
  maximum(data(xs), dims = dims), function (Δ)
    Δ′ = zero(xs)
    _, i = findmax(data(xs), dims = dims)
    Δ′[i] = data(Δ)
    return (nobacksies(:maximum, Δ′),)
  end
end

@grad function minimum(xs;  dims = dims)
  minimum(data(xs),  dims = dims), function (Δ)
    Δ′ = zero(xs)
    _, i = findmin(data(xs),  dims = dims)
    Δ′[i] = data(Δ)
    return (nobacksies(:minimum, Δ′),)
  end
end

# BLAS

LinearAlgebra.diagm(x::TrackedVector) = track(diagm, x)
@grad diagm(x) = diagm(data(x)), Δ -> (diag(Δ),)

x::TrackedMatrix  * y::AbstractMatrix = track(*, x, y)
x::AbstractMatrix * y::TrackedMatrix  = track(*, x, y)
x::TrackedMatrix  * y::TrackedMatrix  = track(*, x, y)

x::TrackedMatrix  * y::AbstractVector = track(*, x, y)
x::AbstractMatrix * y::TrackedVector  = track(*, x, y)
x::TrackedMatrix  * y::TrackedVector  = track(*, x, y)

x::TrackedVector  * y::AbstractVector = track(*, x, y)
x::AbstractVector * y::TrackedVector  = track(*, x, y)
x::TrackedVector  * y::TrackedVector  = track(*, x, y)

@grad a::AbstractMatrix * b::AbstractVecOrMat =
  data(a)*data(b), Δ -> (Δ * transpose(b), transpose(a) * Δ)

# NNlib

using NNlib
import NNlib: softmax, ∇softmax, logsoftmax, ∇logsoftmax,
              conv, ∇conv_data, depthwiseconv, maxpool, meanpool

softmax(xs::TrackedArray) = track(softmax, xs)

@grad softmax(xs) = softmax(data(xs)), Δ -> (nobacksies(:softmax, ∇softmax(data(Δ), data(xs))),)

logsoftmax(xs::TrackedArray) = track(logsoftmax, xs)

@grad logsoftmax(xs) = logsoftmax(data(xs)), Δ -> (nobacksies(:logsoftmax, ∇logsoftmax(data(Δ), data(xs))),)

depthwiseconv(x::TrackedArray, w::TrackedArray; kw...) = track(depthwiseconv, x, w; kw...)
depthwiseconv(x::AbstractArray, w::TrackedArray; kw...) = track(depthwiseconv, x, w; kw...)
depthwiseconv(x::TrackedArray, w::AbstractArray; kw...) = track(depthwiseconv, x, w; kw...)

@grad depthwiseconv(x, w; kw...) =
  depthwiseconv(data(x), data(w); kw...),
    Δ -> nobacksies(:depthwiseconv,
      (NNlib.∇depthwiseconv_data(data.((Δ, x, w))...; kw...),
       NNlib.∇depthwiseconv_filter(data.((Δ, x, w))...; kw...)))

conv(x::TrackedArray,  w::TrackedArray;  kw...) = track(conv, x, w; kw...)
conv(x::AbstractArray, w::TrackedArray;  kw...) = track(conv, x, w; kw...)
conv(x::TrackedArray,  w::AbstractArray; kw...) = track(conv, x, w; kw...)

@grad conv(x, w; kw...) =
  conv(data(x), data(w); kw...),
    Δ -> nobacksies(:conv,
      (NNlib.∇conv_data(data.((Δ, w))...; size=size(x), kw...),
       NNlib.∇conv_filter(data.((Δ, x))..., size(w); kw...)))

∇conv_data(x::TrackedArray,  w::TrackedArray;  kw...) = track(∇conv_data, x, w; kw...)
∇conv_data(x::AbstractArray, w::TrackedArray;  kw...) = track(∇conv_data, x, w; kw...)
∇conv_data(x::TrackedArray,  w::AbstractArray; kw...) = track(∇conv_data, x, w; kw...)

@grad ∇conv_data(x, w; kw...) =
  ∇conv_data(data(x), data(w); kw...),
    Δ -> nobacksies(:conv,
      (NNlib.conv(data.((Δ, w))...; size=size(x), kw...),
       NNlib.∇conv_filter(data.((x, Δ))..., size(w); kw...)))

maxpool(x::TrackedArray, k; kw...) = track(maxpool, x, k; kw...)

@grad function maxpool(x, k; kw...)
  y = maxpool(data(x), k; kw...)
  y, Δ -> (nobacksies(:maxpool, NNlib.∇maxpool(data.((Δ, y, x))..., k; kw...)), nothing)
end

meanpool(x::TrackedArray, k; kw...) = track(meanpool, x, k; kw...)

@grad function meanpool(x, k; kw...)
  y = meanpool(data(x), k; kw...)
  y, Δ -> (nobacksies(:maxpool, NNlib.∇meanpool(data.((Δ, y, x))..., k; kw...)), nothing)
end

# Broadcasting

using ForwardDiff: Dual, partials, value

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
  length(x) == length(Δ) ? trim(x, Δ) :
    trim(x, sum(Δ, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))))

unbroadcast(x::Number, Δ) = sum(Δ)
unbroadcast(x::Base.RefValue, _) = nothing

dual(x, p) = x
dual(x::Real, p) = Dual(x, p)

function partial(f::F, Δ, i, args::Vararg{Any,N}) where {F,N}
  dargs = ntuple(j -> dual(args[j], i==j), Val(N))
  return Δ * f(dargs...).partials[1]
end

@inline function ∇broadcast(f::F, args::Vararg{Any,N}) where {F,N}
  y = broadcast(f, data.(args)...)
  eltype(y) <: Real || return y
  eltype(y) == Bool && return y
  function back(Δ)
    Δargs = ntuple(i -> partial.(f, Δ, i, args...), Val(N))
    dxs = map(unbroadcast, args, Δargs)
    return dxs
  end
  # So we can return non-tracked arrays
  track(Call(back, tracker.(args)), y)
end

using Base.Broadcast: BroadcastStyle, ArrayStyle, Broadcasted, broadcasted

struct TrackedStyle <: BroadcastStyle end

Broadcast.BroadcastStyle(::Type{<:Union{TrackedArray,TrackedReal}}) = TrackedStyle()
Broadcast.BroadcastStyle(::TrackedStyle, ::BroadcastStyle) = TrackedStyle()

# We have to re-build the original broadcast struct to get the appropriate array
# style. We need this primarily to support CuArrays' broadcasting fixes.
broadcast_rebuild(xs) = data(xs)

broadcast_rebuild(bc::Broadcasted) =
  broadcasted(bc.f, broadcast_rebuild.(bc.args)...)

preprocess(x) = x

function Base.Broadcast.materialize(bc::Broadcasted{TrackedStyle})
  bc1 = Broadcast.flatten(bc)
  bc2 = Broadcast.flatten(broadcast_rebuild(bc))
  ∇broadcast(bc2.f, bc1.args...)
end

using Requires

# https://github.com/FluxML/Flux.jl/issues/353
@init Requires.isprecompiling() || @eval Base.Broadcast begin
  function flatten(bc::Broadcasted{Style}) where {Style}
    isflat(bc) && return bc
    args = cat_nested(bc)
    let makeargs = make_makeargs(bc), f = bc.f
      newf = @inline function(args::Vararg{Any,N}) where N
        f(makeargs(args...)...)
      end
      return Broadcasted{Style}(newf, args, bc.axes)
    end
  end
  @inline function make_makeargs(makeargs, t::Tuple{<:Broadcasted,Vararg{Any}})
    bc = t[1]
    let makeargs = make_makeargs(makeargs, tail(t)), f = bc.f
      let makeargs = make_makeargs(makeargs, bc.args)
        headargs, tailargs = make_headargs(bc.args), make_tailargs(bc.args)
        return @inline function(args::Vararg{Any,N}) where N
          args1 = makeargs(args...)
          a, b = headargs(args1...), tailargs(args1...)
          (f(a...), b...)
        end
      end
    end
  end
end
