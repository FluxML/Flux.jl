import Base: *

struct OneHotVector <: AbstractVector{Bool}
  ix::UInt32
  of::UInt32
end

Base.size(xs::OneHotVector) = (Int64(xs.of),)

Base.getindex(xs::OneHotVector, i::Integer) = i == xs.ix

Base.getindex(xs::OneHotVector, ::Colon) = OneHotVector(xs.ix, xs.of)

A::AbstractMatrix * b::OneHotVector = A[:, b.ix]

struct OneHotMatrix{A<:AbstractVector{OneHotVector}} <: AbstractMatrix{Bool}
  height::Int
  data::A
end

Base.size(xs::OneHotMatrix) = (Int64(xs.height),length(xs.data))

Base.getindex(xs::OneHotMatrix, i::Union{Integer, AbstractVector}, j::Integer) = xs.data[j][i]
Base.getindex(xs::OneHotMatrix, ::Colon, i::Integer) = xs.data[i]
Base.getindex(xs::OneHotMatrix, ::Colon, i::AbstractArray) = OneHotMatrix(xs.height, xs.data[i])
Base.getindex(xs::OneHotMatrix, ::Colon, ::Colon) = OneHotMatrix(xs.height, copy(xs.data))

Base.getindex(xs::OneHotMatrix, i::Integer, ::Colon) = map(x -> x[i], xs.data)

A::AbstractMatrix * B::OneHotMatrix = A[:, map(x->x.ix, B.data)]

Base.hcat(x::OneHotVector, xs::OneHotVector...) = OneHotMatrix(length(x), [x, xs...])

batch(xs::AbstractArray{<:OneHotVector}) = OneHotMatrix(length(first(xs)), xs)

import Adapt: adapt, adapt_structure

adapt_structure(T, xs::OneHotMatrix) = OneHotMatrix(xs.height, adapt(T, xs.data))

import .CuArrays: CuArray, cudaconvert
import Base.Broadcast: BroadcastStyle, ArrayStyle
BroadcastStyle(::Type{<:OneHotMatrix{<:CuArray}}) = ArrayStyle{CuArray}()
cudaconvert(x::OneHotMatrix{<:CuArray}) = OneHotMatrix(x.height, cudaconvert(x.data))

"""
    onehot(l, labels[, unk])

Create an [`OneHotVector`](@ref) wtih `l`-th element be `true` based on possible `labels` set.
If `unk` is given, it retruns `onehot(unk, labels)` if the input label `l` is not find in `labels`; otherwise
it will error.

## Examples

```jldoctest
julia> using Flux: onehot

julia> onehot(:b, [:a, :b, :c])
3-element Flux.OneHotVector:
 0
 1
 0

julia> onehot(:c, [:a, :b, :c])
3-element Flux.OneHotVector:
 0
 0
 1
```
"""
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

"""
    onehotbatch(ls, labels[, unk...])

Create an [`OneHotMatrix`](@ref) with a batch of labels based on possible `labels` set, returns the
`onehot(unk, labels)` if given labels `ls` is not found in set `labels`.

## Examples

```jldoctest
julia> using Flux: onehotbatch

julia> onehotbatch([:b, :a, :b], [:a, :b, :c])
3×3 Flux.OneHotMatrix{Array{Flux.OneHotVector,1}}:
 0  1  0
 1  0  1
 0  0  0
```
"""
onehotbatch(ls, labels, unk...) =
  OneHotMatrix(length(labels), [onehot(l, labels, unk...) for l in ls])

Base.argmax(xs::OneHotVector) = xs.ix

"""
    onecold(y[, labels = 1:length(y)])

Inverse operations of [`onehot`](@ref).

```jldoctest
julia> using Flux: onecold

julia> onecold([true, false, false], [:a, :b, :c])
:a

julia> onecold([0.3, 0.2, 0.5], [:a, :b, :c])
:c
```
"""
onecold(y::AbstractVector, labels = 1:length(y)) = labels[Base.argmax(y)]

onecold(y::AbstractMatrix, labels...) =
  dropdims(mapslices(y -> onecold(y, labels...), y, dims=1), dims=1)

onecold(y::OneHotMatrix, labels...) =
  mapreduce(x -> Flux.onecold(x, labels...), |, y.data, dims = 2, init = 0)

# TODO probably still want this as a custom adjoint Zygote
# onecold(x::TrackedVector, l...) = onecold(data(x), l...)
# onecold(x::TrackedMatrix, l...) = onecold(data(x), l...)
