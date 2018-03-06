# Primitive definitions

shape(::typeof(*), A::MatShape{T}, B::VecShape{T}) where T =
  Shape{T}(size(A,1))

shape(::typeof(*), A::MatShape{T}, B::MatShape{T}) where T =
  Shape{T}(size(A,1),size(B,2))

inplace!(::typeof(*), C::AbstractArray, A::AbstractMatrix, B::AbstractArray) =
  A_mul_B!(C, A, B)

shape(::typeof(broadcast), f, xs...) =
  Shape{eltype(xs[1])}(Base.Broadcast.broadcast_shape(size.(xs)...)...)

inplace!(::typeof(broadcast), y, f, xs...) = broadcast!(f, y, xs...)

shape(::typeof(reshape), x::Shape{T}, i...) where T =
  Shape{T}(Base._reshape_uncolon(x, i))

inplace!(::typeof(reshape), y, x, i...) = copy!(y, x)

# NNlib

using NNlib
using ..Tracker: _conv, _maxpool

shape(::typeof(softmax), x) = x
inplace!(::typeof(softmax), y, x) = NNlib.softmax!(y, x)

shape(::typeof(_conv), x::Shape{T}, w::Shape{T}, stride, pad) where T =
  Shape{T}(NNlib.cdims(size(x), size(w), pad, stride))

inplace!(::typeof(_conv), y, x, w, stride, pad) =
  NNlib.conv!(y, x, w, stride = stride, pad = pad)

shape(::typeof(_maxpool), x::Shape{T}, k, pad) where T =
  Shape{T}(NNlib.pdims(size(x), k, pad, k))

inplace!(::typeof(_maxpool), y, x, k, pad) =
  NNlib.maxpool!(y, x, k, pad = pad)
