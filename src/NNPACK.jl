import Flux.Tracker: istracked, back, @back
import NNlib.NNPACK.convo
import NNlib.NNPACK.∇conv_data
import NNlib.NNPACK.∇conv_filter
import NNlib.conv
# import NNlib: NNPACK
# import NNlib: NNPACK.∇conv_data, NNPACK.∇conv_filter

struct Conv{N,F,A,V}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{N,Int}
  dilation::NTuple{N,Int}
end

macro back(x, Δ)
  quote
    x = $(esc(x))
    istracked(x) && back(x, $(esc(Δ)))
  end
end

function _conv(x, w, stride, pad, dilation, activation, bias) 
	println("here 2")
	NNlib.NNPACK.convo(x, w, bias, stride = stride, pad = pad, dilation = dilation, activation = activation)
end

function conv(x, w, bias; stride = 1, pad = 0, dilation = 1, activation = 0)
  println("here 1")
  Tracker.track(_conv, x, w, stride, pad, dilation, activation, bias)
end
# conv(x::AbstractArray{<:Real,N}, w::Tracker.TrackedArray{<:Real,N}, bias; stride = 1, pad = 0, dilation = 1, activation = 0) where N =
#   Tracker.track(_conv, x, w, stride, pad, dilation, activation, bias)
# conv(x::Tracker.TrackedArray{<:Real,N}, w::AbstractArray{<:Real,N}, bias; stride = 1, pad = 0, dilation = 1, activation = 0) where N =
#   Tracker.track(_conv, x, w, stride, pad, dilation, activation, bias)

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

function back(::typeof(_conv), Δ, x, w, stride, pad, dilation, extras...)
  println("here 3")
  @show stride
  @show pad
  @show dilation
  @show extras[2]
  # a = ∇conv_data(Δ, data(x), data(w); stride = stride[1], pad = pad[1], dilation = dilation[1])
  println("here 4")
  # @show a
  @back(x, NNlib.NNPACK.∇conv_data(Δ, data(x), data(w); stride = stride[1], pad = pad[1], dilation = dilation[1], activation = extras[1]))
  @back(w, NNlib.NNPACK.∇conv_filter(Δ, data(x), data(w); stride = stride[1], pad = pad[1], dilation = dilation[1]))
end
