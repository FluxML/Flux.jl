import Base: *

struct OneHotVector{T <: Integer} <: AbstractVector{Bool}
  ix::T
  of::T
end

Base.size(xs::OneHotVector) = (Int(xs.of),)

Base.getindex(xs::OneHotVector, i::Integer) = i == xs.ix

A::AbstractMatrix * b::OneHotVector = A[:, b.ix]

"""
    A matrix of one-hot column vectors
"""
struct OneHotMatrix{A<:AbstractVector{<:Integer}} <: AbstractMatrix{Bool}
  height::Int
  data::A
end

function OneHotMatrix(xs::Vector{<:OneHotVector})
    height = length(xs[1])
    OneHotMatrix(height, map(xs) do x
        length(x) == height || error("All one hot vectors must be the same length")
        x.ix
    end)
end


Base.size(xs::OneHotMatrix) = (xs.height, length(xs.data))

Base.getindex(xs::OneHotMatrix, ::Colon, i::Integer) = OneHotVector(xs.data[i], xs.height)
Base.getindex(xs::OneHotMatrix, i::Integer, j::Integer) = xs[:, j][i]
Base.getindex(xs::OneHotMatrix, ::Colon, i::AbstractArray) = OneHotMatrix(xs.height, xs.data[i])

A::AbstractMatrix * B::OneHotMatrix = A[:, B.data]

Base.hcat(x::OneHotVector, xs::OneHotVector...) = OneHotMatrix([x, xs...])

batch(xs::AbstractArray{<:OneHotVector}) = OneHotMatrix(xs)

import Adapt: adapt, adapt_structure

adapt_structure(T, xs::OneHotMatrix) = OneHotMatrix(xs.height, adapt(T, xs.data))

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  import .CuArrays: CuArray, cudaconvert
  import Base.Broadcast: BroadcastStyle, ArrayStyle
  BroadcastStyle(::Type{<:OneHotMatrix{<:CuArray}}) = ArrayStyle{CuArray}()
  cudaconvert(x::OneHotMatrix{<:CuArray}) = OneHotMatrix(x.height, cudaconvert(x.data))
end

function onehotidx(l, labels)
    i = findfirst(isequal(l), labels)
    i !== nothing || error("Value $(repr(l; context=:limited=>true)) is not in labels")
    i
end

function onehotidx(l, labels, unk)
    i = findfirst(isequal(l), labels)
    i !== nothing || return onehotidx(unk, labels)
    i
end

onehot(l, labels, unk...) = OneHotVector(onhotidx(l, labels, unk...), length(labels))

onehotbatch(ls, labels, unk...) =
  OneHotMatrix(length(labels), [onehotidx(l, labels, unk...) for l in ls])

onecold(y::AbstractVector, labels = 1:length(y)) = labels[Base.argmax(y)]

onecold(y::AbstractMatrix, labels...) =
  dropdims(mapslices(y -> onecold(y, labels...), y, dims=1), dims=1)

function argmax(xs...)
  Base.depwarn("`argmax(...) is deprecated, use `onecold(...)` instead.", :argmax)
  return onecold(xs...)
end

# Ambiguity hack

a::TrackedMatrix * b::OneHotVector = invoke(*, Tuple{AbstractMatrix,OneHotVector}, a, b)
a::TrackedMatrix * b::OneHotMatrix = invoke(*, Tuple{AbstractMatrix,OneHotMatrix}, a, b)

onecold(x::TrackedVector, l...) = onecold(data(x), l...)
onecold(x::TrackedMatrix, l...) = onecold(data(x), l...)
