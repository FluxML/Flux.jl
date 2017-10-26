struct OneHotVector <: AbstractVector{Bool}
  ix::UInt32
  of::UInt32
end

Base.size(xs::OneHotVector) = (Int64(xs.of),)

Base.getindex(xs::OneHotVector, i::Integer) = i == xs.ix

Base.:*(A::AbstractMatrix, b::OneHotVector) = A[:, b.ix]

struct OneHotMatrix{A<:AbstractVector{OneHotVector}} <: AbstractMatrix{Bool}
  height::Int
  data::A
end

Base.size(xs::OneHotMatrix) = (Int64(xs.height),length(xs.data))

Base.getindex(xs::OneHotMatrix, i::Int, j::Int) = xs.data[j][i]

Base.:*(A::AbstractMatrix, B::OneHotMatrix) = A[:, map(x->x.ix, B.data)]

Base.hcat(x::OneHotVector, xs::OneHotVector...) = OneHotMatrix(length(x), [x, xs...])

batch(xs::AbstractArray{<:OneHotVector}) = OneHotMatrix(length(first(xs)), xs)

import NNlib.adapt

adapt(T, xs::OneHotMatrix) = OneHotMatrix(xs.height, adapt(T, xs.data))

@require CuArrays begin
  import CuArrays: CuArray, cudaconvert
  Base.Broadcast._containertype(::Type{<:OneHotMatrix{<:CuArray}}) = CuArray
  cudaconvert(x::OneHotMatrix{<:CuArray}) = OneHotMatrix(x.height, cudaconvert(x.data))
end

function onehot(l, labels)
  i = findfirst(labels, l)
  i > 0 || error("Value $l is not in labels")
  OneHotVector(i, length(labels))
end

onehotbatch(ls, labels) = OneHotMatrix(length(labels), [onehot(l, labels) for l in ls])

argmax(y::AbstractVector, labels = 1:length(y)) =
  labels[findfirst(y, maximum(y))]

argmax(y::AbstractMatrix, l...) =
  squeeze(mapslices(y -> argmax(y, l...), y, 1), 1)
