using NNlib: conv, ∇conv_data, depthwiseconv

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)
"""
    Conv(filter::Tuple, in=>out)
    Conv(filter::Tuple, in=>out, activation)

Standard convolutional layer. `filter` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Example: Applying Conv layer to a 1-channel input using a 2x2 window size,
         giving us a 16-channel output. Output is activated with ReLU.

    filter = (2,2)
    in = 1
    out = 16
    Conv((2, 2), 1=>16, relu)

Data should be stored in WHCN order (width, height, # channels, # batches).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.

Accepts keyword arguments `weight` and `bias` to set the corresponding fields.
Setting `bias` to `Flux.Zeros()` will switch bias off for the layer.

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

"""
    Conv(weight::AbstractArray, bias::AbstractArray)
    Conv(weight::AbstractArray, bias::AbstractArray, activation)

Constructs the convolutional layer with user defined weight and bias arrays.
All other behaviours of the Conv layer apply with regard to data order and
forward pass.

Setting `bias` to `Flux.Zeros()` would switch `bias` off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
"""
function Conv(w::AbstractArray{T,N}, b::Union{Zeros, AbstractVector{T}}, σ = identity;
              stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  return Conv(σ, w, b, stride, pad, dilation)
end

function Conv(;weight::AbstractArray, bias::Union{Zeros, AbstractVector{T}}, activation = identity,
              stride = 1, pad = 0, dilation = 1) where {T,N}
  Conv(weight, bias, activation, stride = stride, pad = pad, dilation = dilation)
end

"""
    convfilter(filter::Tuple, in=>out)

Constructs a standard convolutional weight matrix with given `filter` and
channels from `in` to `out`.

Accepts the keyword `init` (default: `glorot_uniform`) to control the sampling
distribution.

See also: [`depthwiseconvfilter`](@ref)
"""
convfilter(filter::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
          init = glorot_uniform) where N = init(filter..., ch...)

function Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
            init = glorot_uniform,  stride = 1, pad = 0, dilation = 1,
            weight = convfilter(k, ch, init = init), bias = zeros(ch[2])) where N

  Conv(weight, bias, σ,
      stride = stride, pad = pad, dilation = dilation)
end

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
    ConvTranspose(filter::Tuple, in=>out)
    ConvTranspose(filter::Tuple, in=>out, activation)

Standard convolutional transpose layer. `filter` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.

Data should be stored in WHCN order. In other words, a 100×100 RGB image would
be a `100×100×3` array, and a batch of 50 would be a `100×100×3×50` array.

Accepts keyword arguments `weight` and `bias` to set the corresponding fields.
Setting `bias` to `Flux.Zeros()` will switch bias off for the layer.

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

"""
    ConvTranspose(weight::AbstractArray, bias::AbstractArray)
    ConvTranspose(weight::AbstractArray, bias::AbstractArray, activation)

Constructs the convolutional transpose layer with user defined weight and bias arrays.
All other behaviours of the ConvTranspose layer apply with regard to data order and
forward pass.

Setting `bias` to `Flux.Zeros()` would switch `bias` off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
"""
function ConvTranspose(w::AbstractArray{T,N}, b::Union{Zeros, AbstractVector{T}}, σ = identity;
                      stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  return ConvTranspose(σ, w, b, stride, pad, dilation)
end

function ConvTranspose(;weight::AbstractArray{T,N}, bias::Union{Zeros, AbstractVector{T}},
                        activation = identity, stride = 1, pad = 0, dilation = 1) where {T,N}
  ConvTranspose(weight, bias, activation, stride = stride, pad = pad, dilation = dilation)
end

function ConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                      init = glorot_uniform, stride = 1, pad = 0, dilation = 1,
                      weight = convfilter(k, reverse(ch), init = init), bias = zeros(ch[2])) where N
  
  ConvTranspose(weight, bias, σ,
              stride = stride, pad = pad, dilation = dilation)
end

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

function (c::ConvTranspose)(x::AbstractArray)
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  cdims = conv_transpose_dims(c, x)
  σ.(∇conv_data(x, c.weight, cdims) .+ b)
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

"""
    DepthwiseConv(filter::Tuple, in=>out)
    DepthwiseConv(filter::Tuple, in=>out, activation)

Depthwise convolutional layer. `filter` should be a tuple like `(2, 2)`.
`in` and `out` specify the number of input and output channels respectively.
Note that `out` must be an integer multiple of `in`.

Data should be stored in WHCN order. In other words, a 100×100 RGB image would
be a `100×100×3` array, and a batch of 50 would be a `100×100×3×50` array.

Accepts keyword arguments `weight` and `bias` to set the corresponding fields.
Setting `bias` to `Flux.Zeros()` will switch bias off for the layer.

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

"""
    DepthwiseConv(weight::AbstractArray, bias::AbstractArray)
    DepthwiseConv(weight::AbstractArray, bias::AbstractArray, activation)

Constructs the `DepthwiseConv` layer with user defined weight and bias arrays.
All other behaviours of the `DepthwiseConv` layer apply with regard to data order and
forward pass.

Setting `bias` to `Flux.Zeros()` would switch `bias` off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
"""
function DepthwiseConv(w::AbstractArray{T,N}, b::Union{Zeros, AbstractVector{T}}, σ = identity;
                      stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  return DepthwiseConv(σ, w, b, stride, pad, dilation)
end

function DepthwiseConv(;weight::AbstractArray, bias::Union{Zeros, AbstractVector{T}},
                      activation = identity, stride = 1, pad = 0, dilation = 1) where {T,N}
  DepthwiseConv(weight, bias, activation, stride = stride, pad = pad, dilation = dilation)
end

"""
    depthwiseconvfilter(filter::Tuple, in=>out)

Constructs a depthwise convolutional weight array defined by `filter` and channels
from `in` to `out`.

Accepts the keyword `init` (default: `glorot_uniform`) to control the sampling
distribution.

See also: [`convfilter`](@ref)
"""
depthwiseconvfilter(filter::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
                    init = glorot_uniform) where N = init(filter..., div(ch[2], ch[1]), ch[1])

function DepthwiseConv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                      init = glorot_uniform, stride = 1, pad = 0, dilation = 1,
                      weight = depthwiseconvfilter(k, ch, init = init), bias = zeros(ch[2])) where N
  @assert ch[2] % ch[1] == 0 "Output channels must be integer multiple of input channels"

  return DepthwiseConv(
    weight,
    bias,
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

Accepts keyword arguments `weight` and `bias` to set the corresponding fields.
Setting `bias` to `Flux.Zeros()` will switch bias off for the layer.

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

"""
    CrossCor(weight::AbstractArray, bias::AbstractArray)
    CrossCor(weight::AbstractArray, bias::AbstractArray, activation)

Constructs the standard cross convolutional layer with user defined weight and bias
arrays. All other behaviours of the CrossCor layer apply with regard to data order and
forward pass.

Setting `bias` to `Flux.Zeros()` would switch `bias` off for the layer.

Takes the keyword arguments `pad`, `stride` and `dilation`.
"""
function CrossCor(w::AbstractArray{T,N}, b::Union{Zeros, AbstractVector{T}}, σ = identity;
                  stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  return CrossCor(σ, w, b, stride, pad, dilation)
end

function CrossCor(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                  init = glorot_uniform, stride = 1, pad = 0, dilation = 1,
                  weight = convfilter(k, ch, init = init), bias = zeros(ch[2])) where N

  CrossCor(weight, bias, σ,
       stride = stride, pad = pad, dilation = dilation)
end

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
