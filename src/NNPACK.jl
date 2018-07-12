function (c::Conv)(x)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)

  if σ == Flux.relu
  	NNlib.NNPACK.conv(x, c.weight, stride = c.stride, pad = c.pad, dilation = c.dilation, activation = 1, bias = c.bias)
  else
  	σ.(NNlib.NNPACK.conv(x, c.weight, stride = c.stride, pad = c.pad, dilation = c.dilation, activation = 0, bias = c.bias))
  end
end