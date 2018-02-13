# Primitive definitions

inplace!(::typeof(*), C::AbstractArray, A::AbstractMatrix, B::AbstractArray) =
  A_mul_B!(C, A, B)

shape(::typeof(broadcast), f, xs...) =
  Shape{eltype(xs[1])}(Base.Broadcast.broadcast_shape(size.(xs)...)...)

inplace!(::typeof(broadcast), y, f, xs...) = broadcast!(f, y, xs...)
