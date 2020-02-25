using NNlib: conv, ∇conv_data, depthwiseconv, output_size

# pad dims of x with dims of y until ndims(x) == ndims(y)
_paddims(x::Tuple, y::Tuple) = (x..., y[(end - (length(y) - length(x) - 1)):end]...)

_convtransoutdims(isize, ksize, ssize, dsize, pad) = (isize .- 1).*ssize .+ 1 .+ (ksize .- 1).*dsize .- (pad[1:2:end] .+ pad[2:2:end])

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)
"""
    Conv(size, in=>out)
    Conv(size, in=>out, relu)

Standard convolutional layer. `size` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Example: Applying Conv layer to a 1-channel input using a 2x2 window size,
         giving us a 16-channel output. Output is activated with ReLU.

    size = (2,2)
    in = 1
    out = 16
    Conv((2, 2), 1=>16, relu)

Data should be stored in WHCN order (width, height, # channels, batch size).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

Takes the keyword arguments `pad`, `stride` and `dilation`.
"""
struct Conv{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
end

function Conv(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
              stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  return Conv(σ, w, b, stride, pad, dilation)
end

Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
     init = glorot_uniform,  stride = 1, pad = 0, dilation = 1) where N =
  Conv(init(k..., ch...), zeros(ch[2]), σ,
       stride = stride, pad = pad, dilation = dilation)

@functor Conv

function (c::Conv)(x::AbstractArray)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  σ.(conv(x, c.weight, cdims) .+ b)
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
    outdims(l::Conv, isize::Tuple)

Calculate the output dimensions given the input dimensions, `isize`.
Batch size and channel size are ignored as per `NNlib.jl`.

```julia
m = Conv((3, 3), 3 => 16)
outdims(m, (10, 10)) == (8, 8)
outdims(m, (10, 10, 1, 3)) == (8, 8)
```
"""
outdims(l::Conv, isize) =
  output_size(DenseConvDims(_paddims(isize, size(l.weight)), size(l.weight); stride = l.stride, padding = l.pad, dilation = l.dilation))

"""
    ConvTranspose(size, in=>out)
    ConvTranspose(size, in=>out, relu)

Standard convolutional transpose layer. `size` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Data should be stored in WHCN order. In other words, a 100×100 RGB image would
be a `100×100×3` array, and a batch of 50 would be a `100×100×3×50` array.

Takes the keyword arguments `pad`, `stride` and `dilation`.
"""
struct ConvTranspose{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
end

function ConvTranspose(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
              stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  return ConvTranspose(σ, w, b, stride, pad, dilation)
end

ConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
              init = glorot_uniform, stride = 1, pad = 0, dilation = 1) where N =
ConvTranspose(init(k..., reverse(ch)...), zeros(ch[2]), σ,
              stride = stride, pad = pad, dilation = dilation)

@functor ConvTranspose

function conv_transpose_dims(c::ConvTranspose, x::AbstractArray)
    # Calculate size of "input", from ∇conv_data()'s perspective...
    combined_pad = (c.pad[1:2:end] .+ c.pad[2:2:end])
    I = (size(x)[1:end-2] .- 1).*c.stride .+ 1 .+ (size(c.weight)[1:end-2] .- 1).*c.dilation .- combined_pad
    C_in = size(c.weight)[end-1]
    batch_size = size(x)[end]
    # Create DenseConvDims() that looks like the corresponding conv()
    return DenseConvDims((I..., C_in, batch_size), size(c.weight);
        stride=c.stride,
        padding=c.pad,
        dilation=c.dilation,
    )
end

# TODO: Find proper fix for https://github.com/FluxML/Flux.jl/issues/900
@nograd conv_transpose_dims

function (c::ConvTranspose)(x::AbstractArray)
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = conv_transpose_dims(c, x)
  return σ.(∇conv_data(x, c.weight, cdims) .+ b)
end

function Base.show(io::IO, l::ConvTranspose)
  print(io, "ConvTranspose(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)), "=>", size(l.weight, ndims(l.weight)-1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::ConvTranspose{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::ConvTranspose{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

outdims(l::ConvTranspose{N}, isize) where N = _convtransoutdims(isize[1:2], size(l.weight)[1:N], l.stride, l.dilation, l.pad)

"""
    DepthwiseConv(size, in=>out)
    DepthwiseConv(size, in=>out, relu)

Depthwise convolutional layer. `size` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.
Note that `out` must be an integer multiple of `in`.

Data should be stored in WHCN order. In other words, a 100×100 RGB image would
be a `100×100×3` array, and a batch of 50 would be a `100×100×3×50` array.

Takes the keyword arguments `pad`, `stride` and `dilation`.
"""
struct DepthwiseConv{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
end

function DepthwiseConv(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
                       stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  return DepthwiseConv(σ, w, b, stride, pad, dilation)
end

function DepthwiseConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
     init = glorot_uniform, stride = 1, pad = 0, dilation = 1) where N
  @assert ch[2] % ch[1] == 0 "Output channels must be integer multiple of input channels"
  return DepthwiseConv(
    init(k..., div(ch[2], ch[1]), ch[1]),
    zeros(ch[2]),
    σ;
    stride = stride,
    pad = pad,
    dilation = dilation
  )
end

@functor DepthwiseConv

function (c::DepthwiseConv)(x)
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = DepthwiseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  σ.(depthwiseconv(x, c.weight, cdims) .+ b)
end

function Base.show(io::IO, l::DepthwiseConv)
  print(io, "DepthwiseConv(", size(l.weight)[1:end-2])
  print(io, ", ", size(l.weight)[end], "=>", prod(size(l.weight)[end-1:end]))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::DepthwiseConv{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::DepthwiseConv{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

outdims(l::DepthwiseConv, isize) =
  output_size(DepthwiseConvDims(_paddims(isize, (1, 1, size(l.weight)[end], 1)), size(l.weight); stride = l.stride, padding = l.pad, dilation = l.dilation))

"""
    CrossCor(size, in=>out)
    CrossCor(size, in=>out, relu)

Standard cross convolutional layer. `size` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Example: Applying CrossCor layer to a 1-channel input using a 2x2 window size,
         giving us a 16-channel output. Output is activated with ReLU.

    size = (2,2)
    in = 1
    out = 16
    CrossCor((2, 2), 1=>16, relu)

Data should be stored in WHCN order (width, height, # channels, # batches).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

Takes the keyword arguments `pad`, `stride` and `dilation`.
"""
struct CrossCor{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
end

function CrossCor(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
              stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  return CrossCor(σ, w, b, stride, pad, dilation)
end

CrossCor(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
     init = glorot_uniform, stride = 1, pad = 0, dilation = 1) where N =
  CrossCor(init(k..., ch...), zeros(ch[2]), σ,
       stride = stride, pad = pad, dilation = dilation)

@functor CrossCor

function crosscor(x, w, ddims::DenseConvDims)
  ddims = DenseConvDims(ddims, F=true)
  return conv(x, w, ddims)
end

function (c::CrossCor)(x::AbstractArray)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  σ.(crosscor(x, c.weight, cdims) .+ b)
end

function Base.show(io::IO, l::CrossCor)
  print(io, "CrossCor(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::CrossCor{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::CrossCor{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

outdims(l::CrossCor, isize) =
  output_size(DenseConvDims(_paddims(isize, size(l.weight)), size(l.weight); stride = l.stride, padding = l.pad, dilation = l.dilation))

"""
    MaxPool(k)

Max pooling layer. `k` stands for the size of the window for each dimension of the input.

Takes the keyword arguments `pad` and `stride`.
"""
struct MaxPool{N,M}
  k::NTuple{N,Int}
  pad::NTuple{M,Int}
  stride::NTuple{N,Int}
end

function MaxPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N
  stride = expand(Val(N), stride)
  pad = expand(Val(2*N), pad)

  return MaxPool(k, pad, stride)
end

function (m::MaxPool)(x)
    pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return maxpool(x, pdims)
end

function Base.show(io::IO, m::MaxPool)
  print(io, "MaxPool(", m.k, ", pad = ", m.pad, ", stride = ", m.stride, ")")
end

outdims(l::MaxPool{N}, isize) where N = output_size(PoolDims(_paddims(isize, (l.k..., 1, 1)), l.k; stride = l.stride, padding = l.pad))

"""
    MeanPool(k)

Mean pooling layer. `k` stands for the size of the window for each dimension of the input.

Takes the keyword arguments `pad` and `stride`.
"""
struct MeanPool{N,M}
    k::NTuple{N,Int}
    pad::NTuple{M,Int}
    stride::NTuple{N,Int}
end

function MeanPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N
  stride = expand(Val(N), stride)
  pad = expand(Val(2*N), pad)
  return MeanPool(k, pad, stride)
end

function (m::MeanPool)(x)
    pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return meanpool(x, pdims)
end

function Base.show(io::IO, m::MeanPool)
  print(io, "MeanPool(", m.k, ", pad = ", m.pad, ", stride = ", m.stride, ")")
end

outdims(l::MeanPool{N}, isize) where N = output_size(PoolDims(_paddims(isize, (l.k..., 1, 1)), l.k; stride = l.stride, padding = l.pad))