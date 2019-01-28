using NNlib: conv, depthwiseconv

@generated sub2(::Val{N}) where N = :(Val($(N-2)))

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
  Conv(σ, w, b, expand.(sub2(Val(N)), (stride, pad, dilation))...)

Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
     init = glorot_uniform,  stride = 1, pad = 0, dilation = 1) where N =
  Conv(param(init(k..., ch...)), param(zeros(ch[2])), σ,
       stride = stride, pad = pad, dilation = dilation)

@treelike Conv

function (c::Conv)(x::AbstractArray)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  σ.(conv(x, c.weight, stride = c.stride, pad = c.pad, dilation = c.dilation) .+ b)
end

function Base.show(io::IO, l::Conv)
  print(io, "Conv(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::Conv{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::Conv{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

"""
    DepthwiseConv(size, in)
    DepthwiseConv(size, in=>mul)
    DepthwiseConv(size, in=>mul, relu)

Depthwise convolutional layer. `size` should be a tuple like `(2, 2)`.
`in` and `mul` specify the number of input channels and channel multiplier respectively.
In case the `mul` is not specified it is taken as 1.

Data should be stored in WHCN order. In other words, a 100×100 RGB image would
be a `100×100×3` array, and a batch of 50 would be a `100×100×3×50` array.

Takes the keyword arguments `pad` and `stride`.
"""
struct DepthwiseConv{N,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
end

DepthwiseConv(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
       stride = 1, pad = 0) where {T,N} =
  DepthwiseConv(σ, w, b, expand.(sub2(Val(N)), (stride, pad))...)

DepthwiseConv(k::NTuple{N,Integer}, ch::Integer, σ = identity; init = glorot_uniform,
     stride = 1, pad = 0) where N =
  DepthwiseConv(param(init(k..., 1, ch)), param(zeros(ch)), σ,
       stride = stride, pad = pad)

DepthwiseConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; init = glorot_uniform,
     stride::NTuple{N,Integer} = map(_->1,k),
     pad::NTuple{N,Integer} = map(_->0,k)) where N =
  DepthwiseConv(param(init(k..., ch[2], ch[1])), param(zeros(ch[2]*ch[1])), σ,
       stride = stride, pad = pad)

@treelike DepthwiseConv

function (c::DepthwiseConv)(x)
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  σ.(depthwiseconv(x, c.weight, stride = c.stride, pad = c.pad) .+ b)
end

function Base.show(io::IO, l::DepthwiseConv)
  print(io, "DepthwiseConv(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)), "=>", size(l.weight, ndims(l.weight)-1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    MaxPool(k)

Max pooling layer. `k` stands for the size of the window for each dimension of the input.

Takes the keyword arguments `pad` and `stride`.
"""
struct MaxPool{N}
  k::NTuple{N,Int}
  pad::NTuple{N,Int}
  stride::NTuple{N,Int}
end

MaxPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N =
  MaxPool(k, expand(Val(N), pad), expand(Val(N), stride))

(m::MaxPool)(x) = maxpool(x, m.k; pad = m.pad, stride = m.stride)

function Base.show(io::IO, m::MaxPool)
  print(io, "MaxPool(", m.k, ", pad = ", m.pad, ", stride = ", m.stride, ")")
end

"""
    MeanPool(k)

Mean pooling layer. `k` stands for the size of the window for each dimension of the input.

Takes the keyword arguments `pad` and `stride`.
"""
struct MeanPool{N}
    k::NTuple{N,Int}
    pad::NTuple{N,Int}
    stride::NTuple{N,Int}
end

MeanPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N =
  MeanPool(k, expand(Val(N), pad), expand(Val(N), stride))

(m::MeanPool)(x) = meanpool(x, m.k; pad = m.pad, stride = m.stride)

function Base.show(io::IO, m::MeanPool)
  print(io, "MeanPool(", m.k, ", pad = ", m.pad, ", stride = ", m.stride, ")")
end
