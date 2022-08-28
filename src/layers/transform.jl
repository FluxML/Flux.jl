
# "normalise.jl" contains Dropout etc, and Batchnorm etc. 
# Since that file is quite long, WeightNorm is in a new file.

"""
    WeightNorm(layer, weight::Symbol; dims, init=false)

Applies a reparameterisation to parameters in `layer`, such that `layer.weight` is normalised to
```
    layer.weight .* scale ./ norm(layer.weight; dims)
```
before being used by `layer`. The new trainable parameter `scale` has the same shape
as `sum(layer.weight; dims)`, and is initialised to `1`.

Can be stacked: `WeightNorm(WeightNorm(layer, :A; dims=1), :B, dims=1)` applies this
to fields `layer.A` and `layer.B`. We could make `WeightNorm(layer, :A, :B; dims=1)` construct
this, if it turns out to be useful.

Keyword `init=true` is described below.

# Example
```
julia> d1 = Dense(ones(2, 3), [0, 10], identity)
Dense(3 => 2)       # 8 parameters

julia> d1([1,2,3])
2-element Vector{Float64}:
  6.0
 16.0

julia> wd1 = WeightNorm(d1)  # defaults for this layer type
WeightNorm(Dense(3 => 2), :weight; dims=2)

julia> wd1([1,2,3])
2-element Vector{Float64}:
  3.4641016151377553
 13.464101615137755

julia> Flux.params(wd1)  # 3rd entry is new scale parameter
Params([[1.0 1.0 1.0; 1.0 1.0 1.0], [0, 10], [1.0; 1.0;;]])

julia> d2 = Dense(ones(2, 3) ./ sqrt.(sum(abs2, ones(2,3); dims=2)), [0, 10]);

julia> d2([1,2,3])  # same result
2-element Vector{Float64}:
  3.4641016151377553
 13.464101615137755
```

# Lazy Initialisation

The option `init=true` changes the scale and the layer's bias based on the first
minibatch `x` seen. This makes more assumptions about the layer:
* Layer must have a field called `:bias` or `:b`,
  and a field called `:sigma` or `:σ` for the activation function.
* Both the bias and the scale must be mutable arrays.
* `dims` must be chosen such that `length(scale) == size(layer(x))[end-2]`,
  interpreted as the channel dimension.

# Example
```
julia> using Flux, Statistics

julia> n1 = WeightNorm(Dense(3 => 2), init=true)  # no activation function
WeightNorm(Dense(3 => 2), :weight; dims=2, init=true)

julia> x1 = rand(Float32, 3, 10);  # batch of 10 input vectors

julia> y1 = n1(x1);

julia> n1  # has been changed
WeightNorm(Dense(3 => 2), :weight; dims=2)

julia> y1 == n1(x1)  # changes were applied before first output
true

julia> mean(y1, dims=2)  # batch mean is approximately zero
2×1 Matrix{Float32}:
 -9.536743f-8
 -9.536743f-8

julia> std(y1, dims=2)  # batch std is approximately 1
2×1 Matrix{Float32}:
 1.0
 0.9999999

julia> n2 = WeightNorm(Dense(3 => 2, relu), init=true);  # with activation function

julia> mean(n2(x1), dims=2)  # the mean that is 0 is before the relu
2×1 Matrix{Float32}:
 0.40149665
 0.35264373
```

# Reference

Salimans & Kingma, _Weight Normalization_ (2016) https://arxiv.org/abs/1602.07868

"""
struct WeightNorm{field,dims,L,S<:AbstractArray} # <: LazyTransform
  layer::L
  scale::S
  init::Base.RefValue{Bool}  # not implemented yet
end

using Flux
using Flux: functor, @functor
using ChainRulesCore: ignore_derivatives
using Statistics: mean, std

# Flux.@layer WeightNorm
@functor WeightNorm
_big_show(io::IO, w::WeightNorm, indent::Int64) = _layer_show(io, w, indent)

# (w::WeightNorm)(x) = transform(w)(x)
function (w::WeightNorm)(x)
  ignore_derivatives() do
    w.init[] && _weightnorm_init!(w, x)
  end
  transform(w)(x)
end

function transform(w::WeightNorm{which,dims}) where {which,dims}
  fields, re = functor(transform(w.layer))
  done = weightnorm(getfield(fields, which), w.scale, dims)
  re(merge(fields, NamedTuple{(which,)}((done,))))
end

function WeightNorm(layer, which::Symbol; dims, init::Bool=false)
  array = getfield(transform(layer), which)
  array isa AbstractArray{<:Number} || error("WeightNorm must apply to a field which is an array of numbers")
  iszero(array) && @warn "WeightNorm should not be applied to an all-zero array" which string(layer)
  _scale = dims isa Colon ? sum(array; dims=1:ndims(array)) : sum(array; dims)
  scale = map(one, _scale)
  WeightNorm{which, dims, typeof(layer), typeof(scale)}(layer, scale, Ref(init))
end

function weightnorm(array::AbstractArray, scale::AbstractArray, dims)
  n2 = sum(abs2, array; dims)
  ε = eps(eltype(array))
  @. array * scale / sqrt(n2 + ε)
end

"""
    transform(w::WeightNorm)

Takes a normalisation wrapper like `WeightNorm` and returns
the original layer with newly adjusted weights. Recursive.

(Could have a better name? Not necc. specific to WeightNorm.)
"""
transform(x) = x

"""
    _weightnorm_init!(w::WeightNorm, x)

This is called on the first execution, when using `WeightNorm(...; init=true)`.
It mutates `w.scale` and `transform(w).bias`, and sets flag `init=false`.
"""
function _weightnorm_init!(w::WeightNorm{which,dims}, x) where {which,dims}
  # First, make a version of the layer without an activation function:
  layer = transform(w)
  fields, re = functor(layer)
  nt = hasproperty(fields, :σ) ? (σ=identity,) : hasproperty(fields, :sigma) ? (sigma=identity,) : (;)
  noact = re(merge(fields, nt))

  # Second, apply that to the data. The relevant statistics are along all except the channel dims?
  y = noact(x)
  ndims(y) > 1 || begin @error "can't apply _weightnorm_init! to a single sample, need a batch. Doing nothing"; return end
  rdims = ((1:ndims(y)-2)..., ndims(y))
  vec_mean = vec(mean(y; dims=rdims))
  vec_std = vec(std(y; dims=rdims))

  # There is (by default) a scale per channel dimension. 
  if length(w.scale) == length(vec_std)
    vec(w.scale) .= 1 ./ vec_std
  else
    error("WeightNorm's lazy initialisation is confused about what dimensions `scale` corresponds to")
  end

  # There should always be one bias per channel dimension:
  bias = hasproperty(fields, :bias) ? fields.bias : hasproperty(fields, :b) ? fields.b : error("WeightNorm(...; init=true) needs a bias field!")
  bias isa AbstractVector || error("WeightNorm(...; init=true) needs a bias vector!")
  if length(bias) == length(vec_std)
    # This requires an array we can write into. We could change it to make WeightNorm a mutable struct instead.
    bias .= .-vec_mean ./ vec_std
  else
    error("WeightNorm's lazy initialisation is confused about what dimensions bias lines up with")
  end
  
  # Done! We mutated arrays within the underlying layer.
  w.init[] = false
  nothing
end

"""
    WeightNorm(layer)

For known layers, the defaults are as follows:
* Apply normalisation to all weight-like fields, but never to bias
* Choose `dims` so that `scale` has one entry per output channel.

That is, `dims` is chosen to make `length(scale) == size(layer(x))[end-2]` for any
input `x` for which `size(x)[end] == size(layer(x))[end]` is the number of batches.
The same assumption is needed for deferred initialisation via `init=true`.

# Examples

```jldoctest
julia> bi = Flux.Bilinear((3, 3) => 5);

julia> WeightNorm(bi)
WeightNorm(Bilinear(3 => 5), :weight; dims=(2, 3))

julia> bi.weight |> size
(5, 3, 3)

julia> bi(randn(3, 32), randn(3, 32)) |> size  # 5 output channels
(5, 32)

julia> WeightNorm(bi).scale |> length
5
```

Convolutional layers:

```jldoctest
julia> c4 = Conv((4, 4), 3 => 7, pad=1);  # conv layer for 2D colour images.

julia> WeightNorm(c4)
WeightNorm(Conv((4, 4), 3 => 7, pad=1), :weight; dims=(1, 2, 3))

julia> c4.weight |> size
(4, 4, 3, 7)

julia> WeightNorm(c4).scale |> length  # 7 output channels
7
```
"""
WeightNorm(d::Dense; dims=2, kw...) = WeightNorm(d, :weight; dims, kw...)
WeightNorm(d::Bilinear; dims=(2,3), kw...) = WeightNorm(d, :weight; dims, kw...)
WeightNorm(d::Embedding; dims=2, kw...) = WeightNorm(d, :weight; dims, kw...)

WeightNorm(s::Scale; dims=2, kw...) = WeightNorm(s, :scale; dims, kw...)

_conv_wdims(c) = Tuple(1:ndims(c.weight)-1)
WeightNorm(c::Conv; dims=_conv_wdims(c), kw...) = WeightNorm(c, :weight; dims, kw...)
WeightNorm(c::CrossCor; dims=_conv_wdims(c), kw...) = WeightNorm(c, :weight; dims, kw...)
WeightNorm(c::ConvTranspose; dims=_conv_wdims(c), kw...) = WeightNorm(c, :weight; dims, kw...)

# WeightNorm(r::Flux.Recur, args...; kw...) = Flux.Recur(WeightNorm(r.cell, args...; kw...), r.state)
# WeightNorm(r::Flux.RNNCell) =

function WeightNorm(layer)
  train = Optimisers._trainable(layer)::NamedTuple
  list = filter(keys(train)) do s
    s in (:bias, :b, :σ) && return false
    train[s] isa AbstractArray{<:Number} || return false
    return true
  end
  name = typeof(layer).name.name
  isempty(list) && error("WeightNorm found no suitable fields to normalise in $name")
  error("""WeightNorm does now know about laye $name.
           You should probably define a method as like this,
           with some field in $list,
           and some default value for `dims`: 
               Flux.WeightNorm(layer::$name; dims=default, init=false) = WeightNorm(layer, :field; dims, init)""")
end

function Base.show(io::IO, w::WeightNorm{which, dims}) where {which, dims}
  print(io, "WeightNorm(")
  Base.show(io, w.layer)
  print(io, ", :", which, "; dims=", dims)
  if w.init[]
    print(io, ", ")
    printstyled(io, "init=true", color=:magenta)
  end
  print(io, ")")
end
