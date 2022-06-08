using NNlib: conv, ∇conv_data, depthwiseconv, output_size

# pad dims of x with dims of y until ndims(x) == ndims(y)
_paddims(x::Tuple, y::Tuple) = (x..., y[(end - (length(y) - length(x) - 1)):end]...)

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

conv_reshape_bias(c) = conv_reshape_bias(c.bias, c.stride)
conv_reshape_bias(@nospecialize(bias), _) = bias
conv_reshape_bias(bias::AbstractVector, stride) = reshape(bias, map(_->1, stride)..., :, 1)

"""
    SamePad()

Passed as an option to convolutional layers (and friends), this causes
the padding to be chosen such that the input and output sizes agree
(on the first `N` dimensions, the kernel or window) when `stride==1`.
When `stride≠1`, the output size equals `ceil(input_size/stride)`.

See also [`Conv`](@ref), [`MaxPool`](@ref).

# Examples
```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);  # a batch of images

julia> layer = Conv((2,2), 3 => 7, pad=SamePad())
Conv((2, 2), 3 => 7, pad=(1, 0, 1, 0))  # 91 parameters

julia> layer(xs) |> size  # notice how the dimensions stay the same with this padding
(100, 100, 7, 50)

julia> layer2 = Conv((2,2), 3 => 7)
Conv((2, 2), 3 => 7)  # 91 parameters

julia> layer2(xs) |> size  # the output dimension changes as the padding was not "same"
(99, 99, 7, 50)

julia> layer3 = Conv((5, 5), 3 => 7, stride=2, pad=SamePad())
Conv((5, 5), 3 => 7, pad=2, stride=2)  # 532 parameters

julia> layer3(xs) |> size  # output size = `ceil(input_size/stride)` = 50
(50, 50, 7, 50)
```
"""
struct SamePad end

calc_padding(lt, pad, k::NTuple{N,T}, dilation, stride) where {T,N} = expand(Val(2*N), pad)
function calc_padding(lt, ::SamePad, k::NTuple{N,T}, dilation, stride) where {N,T}
  #Ref: "A guide to convolution arithmetic for deep learning" https://arxiv.org/abs/1603.07285

  # Effective kernel size, including dilation
  k_eff = @. k + (k - 1) * (dilation - 1)
  # How much total padding needs to be applied?
  pad_amt = @. k_eff - 1
  # In case amount of padding is odd we need to apply different amounts to each side.
  return Tuple(mapfoldl(i -> [cld(i, 2), fld(i,2)], vcat, pad_amt))
end

"""
    Conv(filter, in => out, σ = identity;
         stride = 1, pad = 0, dilation = 1, groups = 1, [bias, init])

Standard convolutional layer. `filter` is a tuple of integers
specifying the size of the convolutional kernel;
`in` and `out` specify the number of input and output channels.

Image data should be stored in WHCN order (width, height, channels, batch).
In other words, a 100×100 RGB image would be a `100×100×3×1` array,
and a batch of 50 would be a `100×100×3×50` array.
This has `N = 2` spatial dimensions, and needs a kernel size like `(5,5)`,
a 2-tuple of integers.

To take convolutions along `N` feature dimensions, this layer expects as input an array
with `ndims(x) == N+2`, where `size(x, N+1) == in` is the number of input channels,
and `size(x, ndims(x))` is (as always) the number of observations in a batch.
Then:
* `filter` should be a tuple of `N` integers.
* Keywords `stride` and `dilation` should each be either single integer,
  or a tuple with `N` integers.
* Keyword `pad` specifies the number of elements added to the borders of the data array. It can be
  - a single integer for equal padding all around,
  - a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
  - a tuple of `2*N` integers, for asymmetric padding, or
  - the singleton `SamePad()`, to calculate padding such that
    `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.
* Keyword `groups` is expected to be an `Int`. It specifies the number of groups
  to divide a convolution into.

Keywords to control initialization of the layer:
* `init` - Function used to generate initial weights. Defaults to `glorot_uniform`.
* `bias` - The initial bias vector is all zero by default. Trainable bias can be disabled entirely
  by setting this to `false`, or another vector can be provided such as `bias = randn(Float32, out)`.

See also [`ConvTranspose`](@ref), [`DepthwiseConv`](@ref), [`CrossCor`](@ref).

# Examples
```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50); # a batch of images

julia> layer = Conv((5,5), 3 => 7, relu; bias = false)
Conv((5, 5), 3 => 7, relu, bias=false)  # 525 parameters

julia> layer(xs) |> size
(96, 96, 7, 50)

julia> Conv((5,5), 3 => 7; stride = 2)(xs) |> size
(48, 48, 7, 50)

julia> Conv((5,5), 3 => 7; stride = 2, pad = SamePad())(xs) |> size
(50, 50, 7, 50)

julia> Conv((1,1), 3 => 7; pad = (20,10,0,0))(xs) |> size
(130, 100, 7, 50)

julia> Conv((5,5), 3 => 7; stride = 2, dilation = 4)(xs) |> size
(42, 42, 7, 50)
```
"""
struct Conv{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
  groups::Int
end

"""
    Conv(weight::AbstractArray, [bias, activation; stride, pad, dilation])

Constructs a convolutional layer with the given weight and bias.
Accepts the same keywords and has the same defaults as `Conv((4,4), 3 => 7, relu)`.

```jldoctest
julia> weight = rand(3, 4, 5);

julia> bias = zeros(5);

julia> c1 = Conv(weight, bias, sigmoid)  # expects 1 spatial dimension
Conv((3,), 4 => 5, σ)  # 65 parameters

julia> c1(randn(100, 4, 64)) |> size
(98, 5, 64)

julia> Flux.params(c1) |> length
2
```
"""
function Conv(w::AbstractArray{T,N}, b = true, σ = identity;
              stride = 1, pad = 0, dilation = 1, groups = 1) where {T,N}

  @assert size(w, N) % groups == 0 "Output channel dimension must be divisible by groups."
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(Conv, pad, size(w)[1:N-2], dilation, stride)
  bias = create_bias(w, b, size(w, N))
  return Conv(σ, w, bias, stride, pad, dilation, groups)
end

function Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
            init = glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
            bias = true) where N
    
  weight = convfilter(k, ch; init, groups)
  Conv(weight, bias, σ; stride, pad, dilation, groups)
end

"""
    convfilter(filter::Tuple, in => out[; init = glorot_uniform])

Constructs a standard convolutional weight matrix with given `filter` and
channels from `in` to `out`.

Accepts the keyword `init` (default: `glorot_uniform`) to control the sampling
distribution.

This is internally used by the [`Conv`](@ref) layer.
"""
function convfilter(filter::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
          init = glorot_uniform, groups = 1) where N
  cin, cout = ch
  @assert cin % groups == 0 "Input channel dimension must be divisible by groups."
  @assert cout % groups == 0 "Output channel dimension must be divisible by groups."
  init(filter..., cin÷groups, cout)
end

@functor Conv

conv_dims(c::Conv, x::AbstractArray) =
  DenseConvDims(x, c.weight; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)

ChainRulesCore.@non_differentiable conv_dims(::Any, ::Any)

function (c::Conv)(x::AbstractArray)
  σ = NNlib.fast_act(c.σ, x)
  cdims = conv_dims(c, x)
  σ.(conv(x, c.weight, cdims) .+ conv_reshape_bias(c))
end

_channels_in(l::Conv) = size(l.weight, ndims(l.weight)-1) * l.groups
_channels_out(l::Conv) = size(l.weight, ndims(l.weight))

function Base.show(io::IO, l::Conv)
  print(io, "Conv(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", _channels_in(l), " => ", _channels_out(l))
  _print_conv_opt(io, l)
  print(io, ")")
end

function _print_conv_opt(io::IO, l)
  l.σ == identity || print(io, ", ", l.σ)
  all(==(0), l.pad) || print(io, ", pad=", _maybetuple_string(l.pad))
  all(==(1), l.stride) || print(io, ", stride=", _maybetuple_string(l.stride))
  all(==(1), l.dilation) || print(io, ", dilation=", _maybetuple_string(l.dilation))
  if hasproperty(l, :groups)
    (l.groups == 1) || print(io, ", groups=", l.groups)
  end
  (l.bias === false) && print(io, ", bias=false")
end

"""
    ConvTranspose(filter, in => out, σ=identity; stride=1, pad=0, dilation=1, [bias, init])

Standard convolutional transpose layer. `filter` is a tuple of integers
specifying the size of the convolutional kernel, while
`in` and `out` specify the number of input and output channels.

Note that `pad=SamePad()` here tries to ensure `size(output,d) == size(x,d) * stride`.

Parameters are controlled by additional keywords, with defaults
`init=glorot_uniform` and `bias=true`.

See also [`Conv`](@ref) for more detailed description of keywords.

# Examples
```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);  # a batch of 50 RGB images

julia> lay = ConvTranspose((5,5), 3 => 7, relu)
ConvTranspose((5, 5), 3 => 7, relu)  # 532 parameters

julia> lay(xs) |> size
(104, 104, 7, 50)

julia> ConvTranspose((5,5), 3 => 7, stride=2)(xs) |> size
(203, 203, 7, 50)

julia> ConvTranspose((5,5), 3 => 7, stride=3, pad=SamePad())(xs) |> size
(300, 300, 7, 50)
```
"""
struct ConvTranspose{N,M,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
  groups::Int
end

_channels_in(l::ConvTranspose)  = size(l.weight)[end]
_channels_out(l::ConvTranspose) = size(l.weight)[end-1]*l.groups

"""
    ConvTranspose(weight::AbstractArray, [bias, activation; stride, pad, dilation, groups])

Constructs a ConvTranspose layer with the given weight and bias.
Accepts the same keywords and has the same defaults as `ConvTranspose((4,4), 3 => 7, relu)`.

# Examples
```jldoctest
julia> weight = rand(3, 4, 5);

julia> bias = zeros(4);

julia> c1 = ConvTranspose(weight, bias, sigmoid)
ConvTranspose((3,), 5 => 4, σ)  # 64 parameters

julia> c1(randn(100, 5, 64)) |> size  # transposed convolution will increase the dimension size (upsampling)
(102, 4, 64)

julia> Flux.params(c1) |> length
2
```
"""
function ConvTranspose(w::AbstractArray{T,N}, bias = true, σ = identity;
                      stride = 1, pad = 0, dilation = 1, groups=1) where {T,N}
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(ConvTranspose, pad, size(w)[1:N-2], dilation, stride)
  b = create_bias(w, bias, size(w, N-1) * groups)
  return ConvTranspose(σ, w, b, stride, pad, dilation, groups)
end

function ConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                      init = glorot_uniform, stride = 1, pad = 0, dilation = 1,
                      groups = 1,
                      bias = true,
                      ) where N

  weight = convfilter(k, reverse(ch); init, groups)                    
  ConvTranspose(weight, bias, σ; stride, pad, dilation, groups)
end

@functor ConvTranspose

function conv_transpose_dims(c::ConvTranspose, x::AbstractArray)
  # Calculate size of "input", from ∇conv_data()'s perspective...
  combined_pad = (c.pad[1:2:end] .+ c.pad[2:2:end])
  I = (size(x)[1:end-2] .- 1).*c.stride .+ 1 .+ (size(c.weight)[1:end-2] .- 1).*c.dilation .- combined_pad
  C_in = size(c.weight)[end-1] * c.groups
  batch_size = size(x)[end]
  # Create DenseConvDims() that looks like the corresponding conv()
  w_size = size(c.weight)
  return DenseConvDims((I..., C_in, batch_size), w_size;
                      stride=c.stride,
                      padding=c.pad,
                      dilation=c.dilation,
                      groups=c.groups,
  )
end

ChainRulesCore.@non_differentiable conv_transpose_dims(::Any, ::Any)

function (c::ConvTranspose)(x::AbstractArray)
  σ = NNlib.fast_act(c.σ, x)
  cdims = conv_transpose_dims(c, x)
  σ.(∇conv_data(x, c.weight, cdims) .+ conv_reshape_bias(c))
end

function Base.show(io::IO, l::ConvTranspose)
  print(io, "ConvTranspose(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", _channels_in(l), " => ", _channels_out(l))
  _print_conv_opt(io, l)
  print(io, ")")
end

function calc_padding(::Type{ConvTranspose}, pad::SamePad, k::NTuple{N,T}, dilation, stride) where {N,T}
  calc_padding(Conv, pad, k .- stride .+ 1, dilation, stride)
end

"""
    DepthwiseConv(filter, in => out, σ=identity; stride=1, pad=0, dilation=1, [bias, init])
    DepthwiseConv(weight::AbstractArray, [bias, activation; stride, pad, dilation])
    
Return a depthwise convolutional layer, that is a [`Conv`](@ref) layer with number of
groups equal to the number of input channels.

See [`Conv`](@ref) for a description of the arguments.

# Examples

```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);  # a batch of 50 RGB images

julia> lay = DepthwiseConv((5,5), 3 => 6, relu; bias=false)
Conv((5, 5), 3 => 6, relu, groups=3, bias=false)  # 150 parameters 

julia> lay(xs) |> size
(96, 96, 6, 50)

julia> DepthwiseConv((5, 5), 3 => 9, stride=2, pad=2)(xs) |> size
(50, 50, 9, 50)
```
"""
function DepthwiseConv(k::NTuple{<:Any,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; 
            stride = 1, pad = 0, dilation = 1, bias = true, init = glorot_uniform)
  Conv(k, ch, σ; groups=ch.first, stride, pad, dilation, bias, init)
end

function DepthwiseConv(w::AbstractArray{T,N}, bias = true, σ = identity;
                  stride = 1, pad = 0, dilation = 1) where {T,N}
  w2 = reshape(w, size(w)[1:end-2]..., 1, :)
  Conv(w2, bias, σ; groups = size(w)[end-1], stride, pad, dilation)
end


"""
    CrossCor(filter, in => out, σ=identity; stride=1, pad=0, dilation=1, [bias, init])

Standard cross correlation layer. `filter` is a tuple of integers
specifying the size of the convolutional kernel;
`in` and `out` specify the number of input and output channels.

Parameters are controlled by additional keywords, with defaults
`init=glorot_uniform` and `bias=true`.

CrossCor layer can also be manually constructed by passing in weights and
biases. This constructor accepts the layer accepts the same keywords (and has
the same defaults) as the `CrossCor((4,4), 3 => 7, relu)` method.

See also [`Conv`](@ref) for more detailed description of keywords.

# Examples

```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);  # a batch of 50 RGB images

julia> lay = CrossCor((5,5), 3 => 6, relu; bias=false)
CrossCor((5, 5), 3 => 6, relu, bias=false)  # 450 parameters

julia> lay(xs) |> size
(96, 96, 6, 50)

julia> CrossCor((5,5), 3 => 7, stride=3, pad=(2,0))(xs) |> size
(34, 32, 7, 50)
```
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
    CrossCor(weight::AbstractArray, [bias, activation; stride, pad, dilation])

Constructs a CrossCor layer with the given weight and bias.
Accepts the same keywords and has the same defaults as `CrossCor((5,5), 3 => 6, relu)`.

# Examples
```jldoctest
julia> weight = rand(3, 4, 5);

julia> bias = zeros(5);

julia> lay = CrossCor(weight, bias, relu)
CrossCor((3,), 4 => 5, relu)  # 65 parameters

julia> lay(randn(100, 4, 64)) |> size
(98, 5, 64)

julia> Flux.params(lay) |> length
2
```
"""
function CrossCor(w::AbstractArray{T,N}, bias = true, σ = identity;
                  stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(CrossCor, pad, size(w)[1:N-2], dilation, stride)
  b = create_bias(w, bias, size(w, N))
  return CrossCor(σ, w, b, stride, pad, dilation)
end

function CrossCor(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                  init = glorot_uniform, stride = 1, pad = 0, dilation = 1,
                  bias = true) where N

  weight = convfilter(k, ch, init = init)
  return CrossCor(weight, bias, σ; stride, pad, dilation)
end

@functor CrossCor

function crosscor(x, w, ddims::DenseConvDims)
  ddims = DenseConvDims(ddims, F=true)
  return conv(x, w, ddims)
end

crosscor_dims(c::CrossCor, x::AbstractArray) =
  DenseConvDims(x, c.weight; stride = c.stride, padding = c.pad, dilation = c.dilation)

ChainRulesCore.@non_differentiable crosscor_dims(::Any, ::Any)

function (c::CrossCor)(x::AbstractArray)
  σ = NNlib.fast_act(c.σ, x)
  cdims = crosscor_dims(c, x)
  σ.(crosscor(x, c.weight, cdims) .+ conv_reshape_bias(c))
end

function Base.show(io::IO, l::CrossCor)
  print(io, "CrossCor(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), " => ", size(l.weight, ndims(l.weight)))
  _print_conv_opt(io, l)
  print(io, ")")
end

"""
    AdaptiveMaxPool(out::NTuple)

Adaptive max pooling layer. Calculates the necessary window size
such that its output has `size(y)[1:N] == out`.

Expects as input an array with `ndims(x) == N+2`, i.e. channel and
batch dimensions, after the `N` feature dimensions, where `N = length(out)`.

See also [`MaxPool`](@ref), [`AdaptiveMeanPool`](@ref).

# Examples
```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);  # batch of 50 RGB images

julia> AdaptiveMaxPool((25, 25))(xs) |> size
(25, 25, 3, 50)

julia> MaxPool((4,4))(xs) ≈ AdaptiveMaxPool((25, 25))(xs)
true
```
"""
struct AdaptiveMaxPool{S, O}
  out::NTuple{O, Int}
  AdaptiveMaxPool(out::NTuple{O, Int}) where O = new{O + 2, O}(out)
end

function (a::AdaptiveMaxPool{S})(x::AbstractArray{T, S}) where {S, T}
  insize = size(x)[1:end-2]
  outsize = a.out
  stride = insize .÷ outsize
  k = insize .- (outsize .- 1) .* stride
  pad = 0
  pdims = PoolDims(x, k; padding=pad, stride=stride)
  return maxpool(x, pdims)
end

function Base.show(io::IO, a::AdaptiveMaxPool)
  print(io, "AdaptiveMaxPool(", a.out, ")")
end

"""
    AdaptiveMeanPool(out::NTuple)

Adaptive mean pooling layer. Calculates the necessary window size
such that its output has `size(y)[1:N] == out`.

Expects as input an array with `ndims(x) == N+2`, i.e. channel and
batch dimensions, after the `N` feature dimensions, where `N = length(out)`.

See also [`MaxPool`](@ref), [`AdaptiveMaxPool`](@ref).

# Examples
```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);  # batch of 50 RGB images

julia> AdaptiveMeanPool((25, 25))(xs) |> size
(25, 25, 3, 50)

julia> MeanPool((4,4))(xs) ≈ AdaptiveMeanPool((25, 25))(xs)
true
```
"""
struct AdaptiveMeanPool{S, O}
  out::NTuple{O, Int}
  AdaptiveMeanPool(out::NTuple{O, Int}) where O = new{O + 2, O}(out)
end

function (a::AdaptiveMeanPool{S})(x::AbstractArray{T, S}) where {S, T}
  insize = size(x)[1:end-2]
  outsize = a.out
  stride = insize .÷ outsize
  k = insize .- (outsize .- 1) .* stride
  pad = 0
  pdims = PoolDims(x, k; padding=pad, stride=stride)
  return meanpool(x, pdims)
end

function Base.show(io::IO, a::AdaptiveMeanPool)
  print(io, "AdaptiveMeanPool(", a.out, ")")
end

"""
    GlobalMaxPool()

Global max pooling layer.

Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output,
by performing max pooling on the complete (w,h)-shaped feature maps.

See also [`MaxPool`](@ref), [`GlobalMeanPool`](@ref).

```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);

julia> m = Chain(Conv((3,3), 3 => 7), GlobalMaxPool());

julia> m(xs) |> size
(1, 1, 7, 50)

julia> GlobalMaxPool()(rand(3,5,7)) |> size  # preserves 2 dimensions
(1, 5, 7)
```
"""
struct GlobalMaxPool end

function (g::GlobalMaxPool)(x)
  # Input size
  x_size = size(x)
  # Kernel size
  k = x_size[1:end-2]
  # Pooling dimensions
  pdims = PoolDims(x, k)

  return maxpool(x, pdims)
end

function Base.show(io::IO, g::GlobalMaxPool)
  print(io, "GlobalMaxPool()")
end

"""
    GlobalMeanPool()

Global mean pooling layer.

Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output,
by performing mean pooling on the complete (w,h)-shaped feature maps.

```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);

julia> m = Chain(Conv((3,3), 3 => 7), GlobalMeanPool());

julia> m(xs) |> size
(1, 1, 7, 50)
```
"""
struct GlobalMeanPool end

function (g::GlobalMeanPool)(x)
  # Input size
  x_size = size(x)
  # Kernel size
  k = x_size[1:end-2]
  # Pooling dimensions
  pdims = PoolDims(x, k)

  return meanpool(x, pdims)
end

function Base.show(io::IO, g::GlobalMeanPool)
  print(io, "GlobalMeanPool()")
end

"""
    MaxPool(window::NTuple; pad=0, stride=window)

Max pooling layer, which replaces all pixels in a block of
size `window` with one.

Expects as input an array with `ndims(x) == N+2`, i.e. channel and
batch dimensions, after the `N` feature dimensions, where `N = length(window)`.

By default the window size is also the stride in each dimension.
The keyword `pad` accepts the same options as for the `Conv` layer,
including `SamePad()`.

See also [`Conv`](@ref), [`MeanPool`](@ref), [`AdaptiveMaxPool`](@ref), [`GlobalMaxPool`](@ref).

# Examples

```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);  # batch of 50 RGB images

julia> m = Chain(Conv((5, 5), 3 => 7, pad=SamePad()), MaxPool((5, 5), pad=SamePad()))
Chain(
  Conv((5, 5), 3 => 7, pad=2),          # 532 parameters
  MaxPool((5, 5), pad=2),
)

julia> m[1](xs) |> size
(100, 100, 7, 50)

julia> m(xs) |> size
(20, 20, 7, 50)

julia> lay = MaxPool((5,), pad=2, stride=(3,))  # one-dimensional window
MaxPool((5,), pad=2, stride=3)

julia> lay(rand(Float32, 100, 7, 50)) |> size
(34, 7, 50)
```
"""
struct MaxPool{N,M}
  k::NTuple{N,Int}
  pad::NTuple{M,Int}
  stride::NTuple{N,Int}
end

function MaxPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N
  stride = expand(Val(N), stride)
  pad = calc_padding(MaxPool, pad, k, 1, stride)
  return MaxPool(k, pad, stride)
end

function (m::MaxPool)(x)
  pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
  return maxpool(x, pdims)
end

function Base.show(io::IO, m::MaxPool)
  print(io, "MaxPool(", m.k)
  all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
  m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
  print(io, ")")
end

_maybetuple_string(pad) = string(pad)
_maybetuple_string(pad::Tuple) = all(==(pad[1]), pad) ? string(pad[1])  : string(pad)

"""
    MeanPool(window::NTuple; pad=0, stride=window)

Mean pooling layer, averaging all pixels in a block of size `window`.

Expects as input an array with `ndims(x) == N+2`, i.e. channel and
batch dimensions, after the `N` feature dimensions, where `N = length(window)`.

By default the window size is also the stride in each dimension.
The keyword `pad` accepts the same options as for the `Conv` layer,
including `SamePad()`.

See also [`Conv`](@ref), [`MaxPool`](@ref), [`AdaptiveMeanPool`](@ref).

# Examples

```jldoctest
julia> xs = rand(Float32, 100, 100, 3, 50);

julia> m = Chain(Conv((5,5), 3 => 7), MeanPool((5,5), pad=SamePad()))
Chain(
  Conv((5, 5), 3 => 7),                 # 532 parameters
  MeanPool((5, 5), pad=2),
)

julia> m[1](xs) |> size
(96, 96, 7, 50)

julia> m(xs) |> size
(20, 20, 7, 50)
```
"""
struct MeanPool{N,M}
  k::NTuple{N,Int}
  pad::NTuple{M,Int}
  stride::NTuple{N,Int}
end

function MeanPool(k::NTuple{N,Integer}; pad = 0, stride = k) where N
  stride = expand(Val(N), stride)
  pad = calc_padding(MeanPool, pad, k, 1, stride)
  return MeanPool(k, pad, stride)
end

function (m::MeanPool)(x)
  pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
  return meanpool(x, pdims)
end

function Base.show(io::IO, m::MeanPool)
  print(io, "MeanPool(", m.k)
  all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
  m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
  print(io, ")")
end
