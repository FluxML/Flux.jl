"""
    Conv2D(size, in=>out)
    Conv2d(size, in=>out, relu)

Standard convolutional layer. `size` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Data should be stored in WHCN order. In other words, a 100×100 RGB image would
be a `100×100×3` array, and a batch of 50 would be a `100×100×3×50` array.

Takes the keyword arguments `pad` and `stride`.
"""
struct Conv2D{F,A,V}
  σ::F
  weight::A
  bias::V
  stride::Int
  pad::Int
end

Conv2D(w::AbstractArray{T,4}, b::AbstractVector{T}, σ = identity;
       stride = 1, pad = 0) where T =
  Conv2D(σ, w, b, stride, pad)

Conv2D(k::NTuple{2,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
       init = initn, stride = 1, pad = 0) =
  Conv2D(param(init(k..., ch...)), param(zeros(ch[2])), σ, stride = stride, pad = pad)

Flux.treelike(Conv2D)

function (c::Conv2D)(x)
  σ, b = c.σ, reshape(c.bias, 1, 1, :)
  σ.(conv2d(x, c.weight, stride = c.stride, padding = c.pad) .+ b)
end

function Base.show(io::IO, l::Conv2D)
  print(io, "Conv2D((", size(l.weight, 1), ", ", size(l.weight, 2), ")")
  print(io, ", ", size(l.weight, 3), "=>", size(l.weight, 4))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
