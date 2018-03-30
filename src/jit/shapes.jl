using ..Tracker: TrackedArray

struct Shape{T,N}
  dims::NTuple{N,Int}
end

VecShape{T} = Shape{T,1}
MatShape{T} = Shape{T,2}

Shape{T}(dims::Vararg{Integer,N}) where {T,N} = Shape{T,N}(dims)
Shape{T}(dims::NTuple{N,Integer}) where {T,N} = Shape{T,N}(dims)

Base.size(s::Shape) = s.dims
Base.size(s::Shape, n) = s.dims[n]
Base.ndims(s::Shape{T,N}) where {T,N} = N
Base.length(s::Shape) = prod(s.dims)
Base.eltype(s::Shape{T}) where T = T

Base.sizeof(s::Shape{T}) where T = sizeof(T)*prod(size(s))

function Base.show(io::IO, s::Shape{T}) where T
  print(io, "Shape{$T}(")
  join(io, s.dims, ", ")
  print(io, ")")
end

shape(x) = x
shape(x::Shape) = x
shape(x::Tuple) = shape.(x)
shape(x::AbstractArray) = Shape{eltype(x)}(size(x)...)
shape(x::TrackedArray) = shape(x.data)

bytes(s::Shape) = sizeof(s)
bytes(x::Tuple) = sum(bytes.(x))

# Recover structure from byte buffers
# Make sure to hold on to the parent buffer for the lifetime of the data.

function restructure(sh::Shape{T}, buf::Vector{UInt8}) where T
  buf = unsafe_wrap(Array, pointer(buf), sizeof(sh))
  reshape(reinterpret(T, buf), size(sh))
end

# Execution with caches

mutable struct Cached{F,A}
  f::F
  buffer::A
end

function (c::Cached)(args...)
  sh = shape(c.f, shape(args)...)
  bytes(sh) > length(c.buffer) && (c.buffer = similar(c.buffer, bytes(sh)))
  y = restructure(sh, c.buffer)
  inplace!(c.f, y, args...)
end
