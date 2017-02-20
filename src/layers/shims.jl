export Conv2D, MaxPool, Reshape

type Conv2D <: Model
  filter::Param{Array{Float64,4}} # [height, width, inchans, outchans]
  stride::Dims{2}
end

Conv2D(size; in = 1, out = 1, stride = (1,1), init = initn) =
  Conv2D(param(initn(size..., in, out)), stride)

shape(c::Conv2D, in::Dims{2}) =
  (map(i -> (in[i]-size(c.filter,i))÷c.stride[i]+1, (1,2))..., size(c.filter, 4))

shape(c::Conv2D, in::Dims{3}) =
  shape(c, (in[1],in[2]))

type MaxPool <: Model
  size::Dims{2}
  stride::Dims{2}
end

MaxPool(size; stride = (1,1)) =
  MaxPool(size, stride)

shape(c::MaxPool, in::Dims{2}) =
  map(i -> (in[i]-c.size[i])÷c.stride[i]+1, (1,2))

shape(c::MaxPool, in::Dims{3}) =
  (shape(c, (in[1],in[2]))..., in[3])

shape(c::MaxPool, in) = throw(ShapeError(c, in))

immutable Reshape{N}
  dims::Dims{N}
end

Reshape(dims::Integer...) = Reshape(dims)

function shape(r::Reshape, dims)
    prod(dims) == prod(r.dims) || throw(ShapeError(r, dims))
    return r.dims
end

shape(r::Reshape, ::Void) = r.dims
