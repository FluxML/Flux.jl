using NNlib: conv

"""
    Conv(size, in=>out)
    Conv(size, in=>out, relu)

Standard convolutional layer. `size` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Data should be stored in WHCN order. In other words, a 100×100 RGB image would
be a `100×100×3` array, and a batch of 50 would be a `100×100×3×50` array.

Takes the keyword arguments `pad` and `stride`.
"""
struct Conv{N,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

Conv(w::AbstractArray{T}, b::AbstractVector{T}, σ = identity;
       stride = 1, pad = 0) where T =
  Conv(σ, w, b, stride, pad)

Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; init = initn,
     stride::NTuple{N,Integer} = map(_->1,k),
     pad::NTuple{N,Integer} = map(_->0,k)) where N =
  Conv(param(init(k..., ch...)), param(zeros(ch[2])), σ,
       stride = stride, pad = pad)

Flux.treelike(Conv)

function (c::Conv)(x)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  σ.(conv(x, c.weight, stride = c.stride, pad = c.pad) .+ b)
end

function Base.show(io::IO, l::Conv)
  print(io, "Conv(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
