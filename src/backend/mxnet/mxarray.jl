using MXNet

# NDArray is row-major so by default all dimensions are reversed in MXNet.
# MXArray tranposes when loading/storing to fix this.

reversedims!(dest, xs) = permutedims!(dest, xs, ndims(xs):-1:1)

immutable MXArray{N}
  data::mx.NDArray
  scratch::Array{Float32,N}
end

MXArray{T}(data::mx.NDArray, scratch::Array{Float32, T}) = MXArray{T}(data, scratch)

MXArray(data::mx.NDArray) = MXArray(data, Array{Float32}(size(data)))

MXArray(dims::Dims) = MXArray(mx.zeros(reverse(dims)))

Base.size(xs::MXArray) = reverse(size(xs.data))

function Base.copy!(mx::MXArray, xs::AbstractArray)
  @assert size(mx) == size(xs)
  reversedims!(mx.scratch, xs)
  copy!(mx.data, mx.scratch)
  return mx
end

function Base.copy!(xs::AbstractArray, mx::MXArray)
  @assert size(xs) == size(mx)
  copy!(mx.scratch, mx.data)
  reversedims!(xs, mx.scratch)
end

Base.copy(mx::MXArray) = copy!(Array{Float32}(size(mx)), mx)

function MXArray(xs::AbstractArray)
  mx = MXArray(size(xs))
  copy!(mx, xs)
end
