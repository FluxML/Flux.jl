"""
    _handle_batch(f, isize, dimsize)

Gracefully handle ignoring batch dimension.

# Arguments:
- `f`: a function of `isize` (including batch) that computes the output size
- `isize`: the input size as specified by the user
- `dimsize`: the expected number of dimensions for this layer (including batch)
"""
function _handle_batch(f, isize, dimsize)
  indims = length(isize)
  if indims == dimsize
    return f(isize)
  elseif indims == dimsize - 1
    outsize = f((isize..., 1))
    return outsize[1:(end - 1)]
  else
    throw(DimensionMismatch("outdims expects ndims(isize) == $dimsize (got isize = $isize). isize should be the size of the input to the function (with batch size optionally left off)"))
  end
end

# fallback for arbitrary functions/layers
# ideally, users should only rely on this for flatten, etc. inside Chains
"""
    outdims(f, isize)

Calculates the output dimensions of `f(x)` where `size(x) == isize`.
The batch dimension **must** be included.
*Warning: this may be slow depending on `f`*
"""
outdims(f, isize) = size(f(ones(Float32, isize)))

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

_convtransoutdims(isize, ksize, ssize, dsize, pad) =
  (isize .- 1) .* ssize .+ 1 .+ (ksize .- 1) .* dsize .- (pad[1:2:end] .+ pad[2:2:end])

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
    throw(DimensionMismatch("input size should equal ($(size(l.W, 2)), nbatches), got $isize"))

  return _handle_batch(isize -> (size(l.W, 1), Base.tail(isize)...), isize, 2)
end

function outdims(l::Diagonal, isize)
  first(isize) == length(l.α) ||
    throw(DimensionMismatch("input length should equal $(length(l.α)), got $(first(isize))"))

  return _handle_batch(isize -> (length(l.α), Base.tail(isize)...), isize, 2)
end

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
  return _handle_batch(isize -> begin
    cdims = DenseConvDims(isize, size(l.weight);
                          stride = l.stride, padding = l.pad, dilation = l.dilation)
    (output_size(cdims)..., NNlib.channels_out(cdims), isize[end])
  end, isize, ndims(l.weight))

outdims(l::ConvTranspose{N}, isize) where N =
  return _handle_batch(isize -> begin
    cdims = _convtransoutdims(isize[1:(end - 2)], size(l.weight)[1:N], l.stride, l.dilation, l.pad)
    (cdims..., size(l.weight)[end - 1], isize[end])
  end, isize, 4)

outdims(l::DepthwiseConv, isize) =
  return _handle_batch(isize -> begin
    cdims = DepthwiseConvDims(isize, size(l.weight);
                              stride = l.stride, padding = l.pad, dilation = l.dilation)
    (output_size(cdims)..., NNlib.channels_out(cdims), isize[end])
  end, isize, 4)

outdims(l::CrossCor, isize) =
  return _handle_batch(isize -> begin
  cdims = DenseConvDims(isize, size(l.weight);
                        stride = l.stride, padding = l.pad, dilation = l.dilation)
    (output_size(cdims)..., NNlib.channels_out(cdims), isize[end])
  end, isize, 4)

outdims(l::MaxPool{N}, isize) where N =
  return _handle_batch(isize -> begin
    pdims = PoolDims(isize, l.k; stride = l.stride, padding = l.pad)
    (output_size(pdims)..., NNlib.channels_out(pdims), isize[end])
  end, isize, 4)

outdims(l::MeanPool{N}, isize) where N =
  return _handle_batch(isize -> begin
    pdims = PoolDims(isize, l.k; stride = l.stride, padding = l.pad)
    (output_size(pdims)..., NNlib.channels_out(pdims), isize[end])
  end, isize, 4)

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