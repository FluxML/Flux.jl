export Conv, MaxPool

type Conv <: Model
  size::Dims{2}
  features::Int
  stride::Dims{2}
end

Conv(size, features; stride = (1,1)) =
  Conv(size, features, stride)

shape(c::Conv, in::Dims{2}) =
  (map(i -> (in[i]-c.size[i])÷c.stride[i]+1, (1,2))..., c.features)

shape(c::Conv, in::Dims{3}) =
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
