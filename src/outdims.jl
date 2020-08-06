# fallback for arbitrary functions/layers
# since we aren't care about batch dimension, we are free to just set it to 1
"""
    outdims(f, isize)

Calculates the output dimensions of `f(x)` where `size(x) == isize`.
The batch dimension is ignored.
*Warning: this may be slow depending on `f`*
"""
outdims(f, isize) = size(f(ones(Float32, isize..., 1)))[1:end-1]

### start basic ###
"""
    outdims(c::Chain, isize)

Calculate the output dimensions given the input dimensions, `isize`.

```julia
m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32))
outdims(m, (10, 10)) == (6, 6)
```
"""
outdims(c::Chain, isize) = foldr(outdims, reverse(c.layers), init = isize)

"""
outdims(l::Dense, isize)

Calculate the output dimensions given the input dimensions, `isize`.

```julia
m = Dense(10, 5)
outdims(m, (10,)) == (5,)
outdims(m, (10, 2)) == (5, 2)
```
"""
function outdims(l::Dense, isize)
  first(isize) == size(l.W, 2) ||
    throw(DimensionMismatch("input size should equal to ($(size(l.W, 2)), ...), got $isize"))
  return (size(l.W, 1), Base.tail(isize)...)
end

outdims(l::Diagonal, isize) = (length(l.α),)

outdims(l::Maxout, isize) = outdims(first(l.over), isize)

## TODO: SkipConnection

#### end basic ####

#### start conv ####

"""
    outdims(l::Conv, isize::Tuple)

Calculate the output dimensions given the input dimensions `isize`.
Batch size and channel size are ignored as per [NNlib.jl](https://github.com/FluxML/NNlib.jl).

```julia
m = Conv((3, 3), 3 => 16)
outdims(m, (10, 10)) == (8, 8)
outdims(m, (10, 10, 1, 3)) == (8, 8)
```
"""
outdims(l::Conv, isize) =
  output_size(DenseConvDims(_paddims(isize, size(l.weight)), size(l.weight);
                            stride = l.stride, padding = l.pad, dilation = l.dilation))

outdims(l::ConvTranspose{N}, isize) where N =
  _convtransoutdims(isize[1:2], size(l.weight)[1:N], l.stride, l.dilation, l.pad)

outdims(l::DepthwiseConv, isize) =
  output_size(DepthwiseConvDims(_paddims(isize, (1, 1, size(l.weight)[end], 1)), size(l.weight);
                                stride = l.stride, padding = l.pad, dilation = l.dilation))

outdims(l::CrossCor, isize) =
  output_size(DenseConvDims(_paddims(isize, size(l.weight)), size(l.weight);
                            stride = l.stride, padding = l.pad, dilation = l.dilation))

outdims(l::MaxPool{N}, isize) where N =
  output_size(PoolDims(_paddims(isize, (l.k..., 1, 1)), l.k; stride = l.stride, padding = l.pad))

outdims(l::MeanPool{N}, isize) where N =
  output_size(PoolDims(_paddims(isize, (l.k..., 1, 1)), l.k; stride = l.stride, padding = l.pad))

## TODO: global and adaptive pooling

#### end conv ####

#### start normalise ####

"""
    outdims(::Dropout, isize)
    outdims(::AlphaDropout, isize)
    outdims(::LayerNorm, isize)
    outdims(::BatchNorm, isize)
    outdims(::InstanceNorm, isize)
    outdims(::GroupNorm, isize)

Calculate the output dimensions given the input dimensions, `isize`.
For a these layers, `outdims(layer, isize) == isize`.

*Note*: since normalisation layers do not store the input size info,
  `isize` is directly returned with no dimension checks.
These definitions exist for convenience.
"""
outdims(::Dropout, isize) = isize
outdims(::AlphaDropout, isize) = isize
outdims(::LayerNorm, isize) = isize
outdims(::BatchNorm, isize) = isize
outdims(::InstanceNorm, isize) = isize
outdims(::GroupNorm, isize) = isize

#### end normalise ####