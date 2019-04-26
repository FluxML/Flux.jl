import Base: *

struct OneHotVector <: AbstractVector{Bool}
  ix::UInt32
  of::UInt32
end

Base.size(xs::OneHotVector) = (Int64(xs.of),)

Base.getindex(xs::OneHotVector, i::Integer) = i == xs.ix

A::AbstractMatrix * b::OneHotVector = A[:, b.ix]

struct OneHotMatrix{A<:AbstractVector{OneHotVector}} <: AbstractMatrix{Bool}
  height::Int
  data::A
end

Base.size(xs::OneHotMatrix) = (Int64(xs.height),length(xs.data))

Base.getindex(xs::OneHotMatrix, i::Integer, j::Integer) = xs.data[j][i]
Base.getindex(xs::OneHotMatrix, ::Colon, i::Integer) = xs.data[i]
Base.getindex(xs::OneHotMatrix, ::Colon, i::AbstractArray) = OneHotMatrix(xs.height, xs.data[i])

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

"""
    onehot(l, labels[, unk])

Create an [`OneHotVector`](@ref) wtih `l`-th element be `true` based on possible `labels` set.
If `unk` is given, it retruns `onehot(unk, labels)` if the input label `l` is not find in `labels`; otherwise
it will error.

## Examples

```jldoctest
julia> onehot(:b, [:a, :b, :c])
3-element Flux.OneHotVector:
 false
  true
 false

julia> onehot(:c, [:a, :b, :c])
3-element Flux.OneHotVector:
 false
 false
  true
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
julia> onehotbatch([:b, :a, :b], [:a, :b, :c])
3×3 Flux.OneHotMatrix:
 false   true  false
  true  false   true
 false  false  false

```
"""
onehotbatch(ls, labels, unk...) =
  OneHotMatrix(length(labels), [onehot(l, labels, unk...) for l in ls])

"""
    onecold(y[, labels = 1:length(y)])

Inverse operations of [`onehot`](@ref).

## Examples

```jldoctest
julia> onecold([true, false, false], [:a, :b, :c])
:a

julia> onecold([0.3, 0.2, 0.5], [:a, :b, :c])
:c
```
"""
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
