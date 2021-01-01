import Base: *

struct OneHotVector <: AbstractVector{Bool}
  ix::UInt32
  of::UInt32
  OneHotVector(ix, of) = 0 < ix <= of ? new(ix, of) :
  throw(DimensionMismatch("index must be between 1 and vector size"))
end

Base.size(xs::OneHotVector) = (Int64(xs.of),)

Base.getindex(xs::OneHotVector, i::Integer) = i == xs.ix

Base.getindex(xs::OneHotVector, ::Colon) = OneHotVector(xs.ix, xs.of)

function Base.:*(A::AbstractMatrix, b::OneHotVector)
  if size(A, 2) != b.of
    throw(DimensionMismatch("Matrix column must correspond with OneHotVector size"))
  end
  return A[:, b.ix]
end

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

# remove workaround when https://github.com/JuliaGPU/CuArrays.jl/issues/676 is fixed
A::AbstractMatrix * B::OneHotMatrix = A[:, cpu(map(x->x.ix, B.data))]

Base.hcat(x::OneHotVector, xs::OneHotVector...) = OneHotMatrix(length(x), [x, xs...])

batch(xs::AbstractArray{<:OneHotVector}) = OneHotMatrix(length(first(xs)), xs)

import Adapt: adapt, adapt_structure

adapt_structure(T, xs::OneHotMatrix) = OneHotMatrix(xs.height, adapt(T, xs.data))

import .CUDA: CuArray, CuArrayStyle, cudaconvert
import Base.Broadcast: BroadcastStyle, ArrayStyle
BroadcastStyle(::Type{<:OneHotMatrix{<:CuArray}}) = CuArrayStyle{2}()
cudaconvert(x::OneHotMatrix{<:CuArray}) = OneHotMatrix(x.height, cudaconvert(x.data))

"""
    onehot(l, labels[, unk])

Return a `OneHotVector` where only first occourence of `l` in `labels` is `1` and
all other elements are `0`.

If `l` is not found in labels and  `unk` is present, the function returns
`onehot(unk, labels)`; otherwise the function raises an error.

# Examples
```jldoctest
julia> Flux.onehot(:b, [:a, :b, :c])
3-element Flux.OneHotVector:
 0
 1
 0

julia> Flux.onehot(:c, [:a, :b, :c])
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

Return a `OneHotMatrix` where `k`th column of the matrix is `onehot(ls[k], labels)`.

If one of the input labels `ls` is not found in `labels` and `unk` is given,
return [`onehot(unk, labels)`](@ref) ; otherwise the function will raise an error.

# Examples
```jldoctest
julia> Flux.onehotbatch([:b, :a, :b], [:a, :b, :c])
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

# Examples
```jldoctest
julia> Flux.onecold([true, false, false], [:a, :b, :c])
:a

julia> Flux.onecold([0.3, 0.2, 0.5], [:a, :b, :c])
:c
```
"""
onecold(y::AbstractVector, labels = 1:length(y)) = labels[Base.argmax(y)]

onecold(y::AbstractMatrix, labels...) =
  dropdims(mapslices(y -> onecold(y, labels...), y, dims=1), dims=1)

onecold(y::OneHotMatrix, labels...) = map(x -> Flux.onecold(x, labels...), y.data)

@nograd onecold, onehot, onehotbatch
