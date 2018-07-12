using NNlib: conv

@generated sub2(::Type{Val{N}}) where N = :(Val{$(N-2)})

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

"""
    Conv(size, in=>out)
    Conv(size, in=>out, relu)

Standard convolutional layer. `size` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Data should be stored in WHCN order. In other words, a 100×100 RGB image would
be a `100×100×3` array, and a batch of 50 would be a `100×100×3×50` array.

Takes the keyword arguments `pad`, `stride` and `dilation`.
"""
struct Conv{N,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
  dilation::NTuple{N,Int}
end

Conv(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
     stride = 1, pad = 0, dilation = 1) where {T,N} =
  Conv(σ, w, b, expand.(sub2(Val{N}), (stride, pad, dilation))...)

Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; init = initn,
     stride = 1, pad = 0, dilation = 1) where N =
  Conv(param(init(k..., ch...)), param(zeros(Float32, ch[2])), σ,
       stride = stride, pad = pad, dilation = dilation)

Flux.treelike(Conv)

function (c::Conv)(x)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  # @show typeof(x)
  # @show typeof(c.weight)
  # @show typeof(conv(x, c.weight, stride = c.stride, pad = c.pad, dilation = c.dilation) .+ b)
  # @show typeof(σ.(conv(x, c.weight, stride = c.stride, pad = c.pad, dilation = c.dilation) .+ b))
  σ.(conv(x, c.weight, stride = c.stride, pad = c.pad, dilation = c.dilation) .+ b)
end

function Base.show(io::IO, l::Conv)
  print(io, "Conv(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end
