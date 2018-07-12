using NNlib: NNPACK

struct Conv{N,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
  dilation::NTuple{N,Int}
end

function (c::Conv)(x)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  wt = copy(c.weight.data)
  # @show σ
  if σ == NNlib.relu
  	# println("Here")
  	NNlib.NNPACK.convo(x, wt, c.bias, stride = c.stride, pad = c.pad, dilation = c.dilation, activation = 1)
  else
  	println("There")
  	σ.(NNlib.NNPACK.convo(x, wt, c.bias, stride = c.stride, pad = c.pad, dilation = c.dilation, activation = 0))
  end
end