import NNlib.conv
using NNlib: NNPACK
using Flux: Tracker

struct Conv{N,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
  dilation::NTuple{N,Int}
end

_conv(x, w, stride, pad, dilation, activation, bias) = NNlib.NNPACK.convo(x, w, bias, stride = stride, pad = pad, dilation = dilation, activation = activation)

conv(x::TrackedArray{<:Real,N}, w::TrackedArray{<:Real,N}, bias; stride = 1, pad = 0, dilation = 1, activation = 0) where N =
  Tracker.track(_conv, x, w, stride, pad, dilation, activation, bias)
conv(x::AbstractArray{<:Real,N}, w::TrackedArray{<:Real,N}, bias; stride = 1, pad = 0, dilation = 1, activation = 0) where N =
  Tracker.track(_conv, x, w, stride, pad, dilation, activation, bias)
conv(x::TrackedArray{<:Real,N}, w::AbstractArray{<:Real,N}, bias; stride = 1, pad = 0, dilation = 1, activation = 0) where N =
  Tracker.track(_conv, x, w, stride, pad, dilation, activation, bias)

function (c::Conv)(x)
  # TODO: breaks gpu broadcast :(
  # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  # @show σ
  if σ == NNlib.relu
  	# println("Here")
  	conv(x, c.weight, c.bias, stride = c.stride, pad = c.pad, dilation = c.dilation, activation = 1)
  else
  	println("There")
  	σ.(conv(x, c.weight, c.bias, stride = c.stride, pad = c.pad, dilation = c.dilation, activation = 0))
  end
end


function back(::typeof(_conv), Δ, x, w, stride, pad, dilation)
  Tracker.@back(x, NNlib.NNPACK.∇conv_data(Δ, data(x), data(w); stride = stride, pad = pad, dilation = dilation))
  Tracker.@back(w, NNlib.NNPACK.∇conv_filter(Δ, data(x), data(w); stride = stride, pad = pad, dilation = dilation))
end