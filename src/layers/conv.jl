"""
    Conv2D(size, in=>out)
    Conv2d(size, in=>out, relu)

Standard convolutional layer. `size` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Data should be stored in HWCN order. In other words, a 100×100 RGB image would
be a `100×100×3` array, and a batch of 50 would be a `100×100×3×50` array.

Takes the keyword arguments `pad` and `stride`.
"""
struct Conv2D{F,A}
  σ::F
  weight::A
  stride::Int
  pad::Int
end

Conv2D(k::NTuple{2,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
       init = initn, stride = 1, pad = 0) =
  Conv2D(σ, param(initn(k..., ch...)), stride, pad)

Flux.treelike(Conv2D)

(c::Conv2D)(x) = c.σ.(conv2d(x, c.weight, stride = c.stride, padding = c.pad))

function Base.show(io::IO, l::Conv2D)
  print(io, "Conv2D((", size(l.weight, 1), ", ", size(l.weight, 2), ")")
  print(io, ", ", size(l.weight, 3), "=>", size(l.weight, 4))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
