import Base: *

struct OneHotVector <: AbstractVector{Bool}
  ix::UInt32
  of::UInt32
end

Base.size(xs::OneHotVector) = (Int64(xs.of),)

Base.getindex(xs::OneHotVector, i::Integer) = i == xs.ix

Base.getindex(xs::OneHotVector, ::Colon) = xs

A::AbstractMatrix * b::OneHotVector = A[:, b.ix]

struct OneHotMatrix{A<:AbstractVector{OneHotVector}} <: AbstractMatrix{Bool}
  height::Int
  data::A
end

Base.size(xs::OneHotMatrix) = (Int64(xs.height),length(xs.data))

Base.getindex(xs::OneHotMatrix, i::Integer, j::Integer) = xs.data[j][i]
Base.getindex(xs::OneHotMatrix, ::Colon, i::Integer) = xs.data[i]
Base.getindex(xs::OneHotMatrix, ::Colon, i::AbstractArray) = OneHotMatrix(xs.height, xs.data[i])

Base.getindex(xs::Flux.OneHotMatrix, j::Base.UnitRange, i::Int) = xs.data[i][j]

Base.getindex(xs::OneHotMatrix, ::Colon, ::Colon) = xs

# handle special case for when we want the entire column without allocating
function Base.getindex(xs::Flux.OneHotMatrix{T}, ot::Union{Base.Slice, Base.OneTo}, i::Int) where {T<:AbstractArray}
  res = similar(xs, size(xs, 1), 1)
  if length(ot) == size(xs, 1)
    res = xs[:,i]
  else
    res = xs[1:length(ot),i]
  end
  res
end

A::AbstractMatrix * B::OneHotMatrix = A[:, map(x->x.ix, B.data)]

Base.hcat(x::OneHotVector, xs::OneHotVector...) = OneHotMatrix(length(x), [x, xs...])

batch(xs::AbstractArray{<:OneHotVector}) = OneHotMatrix(length(first(xs)), xs)

import Adapt: adapt, adapt_structure

adapt_structure(T, xs::OneHotMatrix) = OneHotMatrix(xs.height, adapt(T, xs.data))

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  import .CuArrays: CuArray, cudaconvert
  import Base.Broadcast: BroadcastStyle, ArrayStyle
  BroadcastStyle(::Type{<:OneHotMatrix{<:CuArray}}) = ArrayStyle{CuArray}()
  cudaconvert(x::OneHotMatrix{<:CuArray}) = OneHotMatrix(x.height, cudaconvert(x.data))
end

function onehot(l, labels)
  i = something(findfirst(isequal(l), labels), 0)
  i > 0 || error("Value $l is not in labels")
  OneHotVector(i, length(labels))
end

function onehot(l, labels, unk)
  i = something(findfirst(isequal(l), labels), 0)
  i > 0 || return onehot(unk, labels)
  OneHotVector(i, length(labels))
end

onehotbatch(ls, labels, unk...) =
  OneHotMatrix(length(labels), [onehot(l, labels, unk...) for l in ls])

Base.argmax(xs::OneHotVector) = xs.ix

onecold(y::AbstractVector, labels = 1:length(y)) = labels[Base.argmax(y)]

onecold(y::AbstractMatrix, labels...) =
  dropdims(mapslices(y -> onecold(y, labels...), y, dims=1), dims=1)

function argmax(xs...)
  Base.depwarn("`argmax(...)` is deprecated, use `onecold(...)` instead.", :argmax)
  return onecold(xs...)
end

# Ambiguity hack

a::TrackedMatrix * b::OneHotVector = invoke(*, Tuple{AbstractMatrix,OneHotVector}, a, b)
a::TrackedMatrix * b::OneHotMatrix = invoke(*, Tuple{AbstractMatrix,OneHotMatrix}, a, b)

onecold(x::TrackedVector, l...) = onecold(data(x), l...)
onecold(x::TrackedMatrix, l...) = onecold(data(x), l...)
