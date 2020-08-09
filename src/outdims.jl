"""
    _handle_batch(f, isize, dimsize)

Gracefully handle ignoring batch dimension.

# Arguments:
- `f`: a function of `isize` (including batch) that computes the output size
- `isize`: the input size as specified by the user
- `dimsize`: the expected number of dimensions for this layer (including batch)
- `preserve_batch`: set to `true` to always retain the batch dimension
"""
function _handle_batch(f, isize, dimsize; preserve_batch = false)
  indims = length(isize)
  if indims == dimsize
    return f(isize)
  elseif indims == dimsize - 1
    outsize = f((isize..., 1))
    return preserve_batch ? outsize : outsize[1:(end - 1)]
  else
    throw(DimensionMismatch("outdims expects ndims(isize) == $dimsize (got isize = $isize). isize should be the size of the input to the function (with batch size optionally left off)"))
  end
end

# fallback for arbitrary functions/layers
# ideally, users should only rely on this for flatten, etc. inside Chains
"""
    outdims(f, isize...)

Calculates the output dimensions of `f(x)` where `size(x) == isize`.
The batch dimension **must** be included.
*Warning: this may be slow depending on `f`*
"""
outdims(f, isize...; preserve_batch = false) = size(f([ones(Float32, s) for s in isize]...))

### start basic ###
"""
    outdims(c::Chain, isize)

Calculate the output dimensions given the input dimensions, `isize`.

```julia
m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32))
outdims(m, (10, 10)) == (6, 6)
```
"""
function outdims(c::Chain, isize; preserve_batch = false)
  # if the first layer has different output with
  # preserve_batch = true vs preserve_batch = false
  # then the batch dimension is not included by the user
  initsize = outdims(first(c.layers), isize; preserve_batch = true)
  hasbatch = (outdims(first(c.layers), isize) == initsize)
  outsize = foldl((isize, layer) -> outdims(layer, isize; preserve_batch = true),
                  tail(c.layers); init = initsize)
  
  return hasbatch ? outsize : outsize[1:(end - 1)]
end

"""
outdims(l::Dense, isize; preserve_batch = false)

Calculate the output dimensions given the input dimensions, `isize`.
Set `preserve_batch` to `true` to always return with the batch dimension included.

```julia
m = Dense(10, 5)
outdims(m, (10,)) == (5,)
outdims(m, (10, 2)) == (5, 2)
```
"""
function outdims(l::Dense, isize; preserve_batch = false)
  first(isize) == size(l.W, 2) ||
    throw(DimensionMismatch("input size should equal ($(size(l.W, 2)), nbatches), got $isize"))

  return _handle_batch(isize -> (size(l.W, 1), Base.tail(isize)...), isize, 2; preserve_batch = preserve_batch)
end

function outdims(l::Diagonal, isize; preserve_batch = false)
  first(isize) == length(l.α) ||
    throw(DimensionMismatch("input length should equal $(length(l.α)), got $(first(isize))"))

  return _handle_batch(isize -> (length(l.α), Base.tail(isize)...), isize, 2; preserve_batch = preserve_batch)
end

outdims(l::Maxout, isize; preserve_batch = false) = outdims(first(l.over), isize; preserve_batch = preserve_batch)

function outdims(l::SkipConnection, isize; preserve_batch = false)
  branch_outsize = outdims(l.layers, isize; preserve_batch = preserve_batch)

  return outdims(l.connection, branch_outsize, isize; preserve_batch = preserve_batch)
end

#### end basic ####

#### start conv ####

_convtransoutdims(isize, ksize, ssize, dsize, pad) =
  (isize .- 1) .* ssize .+ 1 .+ (ksize .- 1) .* dsize .- (pad[1:2:end] .+ pad[2:2:end])

"""
    outdims(l::Conv, isize; preserve_batch = false)

Calculate the output dimensions given the input dimensions `isize`.
Set `preserve_batch` to `true` to always return with the batch dimension included.

```julia
m = Conv((3, 3), 3 => 16)
outdims(m, (10, 10)) == (8, 8)
outdims(m, (10, 10, 1, 3)) == (8, 8)
```
"""
outdims(l::Conv, isize; preserve_batch = false) =
  return _handle_batch(isize -> begin
    cdims = DenseConvDims(isize, size(l.weight);
                          stride = l.stride, padding = l.pad, dilation = l.dilation)
    (output_size(cdims)..., NNlib.channels_out(cdims), isize[end])
  end, isize, ndims(l.weight); preserve_batch = preserve_batch)

outdims(l::ConvTranspose{N}, isize; preserve_batch = false) where N =
  return _handle_batch(isize -> begin
    cdims = _convtransoutdims(isize[1:(end - 2)], size(l.weight)[1:N], l.stride, l.dilation, l.pad)
    (cdims..., size(l.weight)[end - 1], isize[end])
  end, isize, 4; preserve_batch = preserve_batch)

outdims(l::DepthwiseConv, isize; preserve_batch = false) =
  return _handle_batch(isize -> begin
    cdims = DepthwiseConvDims(isize, size(l.weight);
                              stride = l.stride, padding = l.pad, dilation = l.dilation)
    (output_size(cdims)..., NNlib.channels_out(cdims), isize[end])
  end, isize, 4; preserve_batch = preserve_batch)

outdims(l::CrossCor, isize; preserve_batch = false) =
  return _handle_batch(isize -> begin
  cdims = DenseConvDims(isize, size(l.weight);
                        stride = l.stride, padding = l.pad, dilation = l.dilation)
    (output_size(cdims)..., NNlib.channels_out(cdims), isize[end])
  end, isize, 4; preserve_batch = preserve_batch)

outdims(l::MaxPool{N}, isize; preserve_batch = false) where N =
  return _handle_batch(isize -> begin
    pdims = PoolDims(isize, l.k; stride = l.stride, padding = l.pad)
    (output_size(pdims)..., NNlib.channels_out(pdims), isize[end])
  end, isize, 4; preserve_batch = preserve_batch)

outdims(l::MeanPool{N}, isize; preserve_batch = false) where N =
  return _handle_batch(isize -> begin
    pdims = PoolDims(isize, l.k; stride = l.stride, padding = l.pad)
    (output_size(pdims)..., NNlib.channels_out(pdims), isize[end])
  end, isize, 4; preserve_batch = preserve_batch)

outdims(l::AdaptiveMaxPool, isize; preserve_batch = false) =
  return _handle_batch(isize -> (l.out..., isize[end - 1], isize[end]),
                       isize, 4; preserve_batch = preserve_batch)

outdims(l::AdaptiveMeanPool, isize; preserve_batch = false) =
  return _handle_batch(isize -> (l.out..., isize[end - 1], isize[end]),
                       isize, 4; preserve_batch = preserve_batch)

outdims(::GlobalMaxPool, isize; preserve_batch = false) =
  return _handle_batch(isize -> (1, 1, isize[end - 1], isize[end]),
                       isize, 4; preserve_batch = preserve_batch)

outdims(::GlobalMeanPool, isize; preserve_batch = false) =
  return _handle_batch(isize -> (1, 1, isize[end - 1], isize[end]),
                       isize, 4; preserve_batch = preserve_batch)

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