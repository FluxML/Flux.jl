using MXNet

# NDArray is row-major so by default all dimensions are reversed in MXNet.
# MXArray tranposes when loading/storing to fix this.

reversedims!(dest, xs) = permutedims!(dest, xs, ndims(xs):-1:1)

struct MXArray{N}
  data::mx.NDArray
  scratch::Array{Float32,N}
end

MXArray(data::mx.NDArray) = MXArray(data, Array{Float32}(size(data)))

# TODO: split cpu/gpu mxarrays
MXArray(dims::Dims, ctx = mx.cpu()) = MXArray(mx.zeros(reverse(dims), ctx))

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

function MXArray(xs::AbstractArray, ctx = mx.cpu())
  mx = MXArray(size(xs), ctx)
  copy!(mx, xs)
end

Base.setindex!(xs::MXArray, x::Real, ::Colon) = xs.data[:] = x
