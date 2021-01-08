module NilNumber

using NNlib
import Random

"""
    Nil <: Number

Nil is a singleton type with a single instance `nil`.
Unlike `Nothing` and `Missing` it subtypes `Number`.
"""
struct Nil <: Number end

const nil = Nil()

Nil(::T) where T<:Number = nil
(::Type{T})(::Nil) where T<:Number = nil
Base.convert(::Type{Nil}, ::Number) = nil

Base.float(::Type{Nil}) = Nil

for f in [:copy, :zero, :one, :oneunit,
          :+, :-, :abs, :abs2, :inv,
          :exp, :log, :log1p, :log2, :log10,
          :sqrt, :tanh, :conj]
  @eval Base.$f(::Nil) = nil
end

for f in [:+, :-, :*, :/, :^, :mod, :div, :rem]
  @eval Base.$f(::Nil, ::Nil) = nil
end

Base.isless(::Nil, ::Nil) = true
Base.isless(::Nil, ::Number) = true
Base.isless(::Number, ::Nil) = true

Base.isnan(::Nil) = false

Base.typemin(::Type{Nil}) = nil
Base.typemax(::Type{Nil}) = nil

Base.promote_rule(x::Type{Nil}, y::Type{<:Number}) = Nil

Random.rand(rng::Random.AbstractRNG, ::Random.SamplerType{Nil}) = Nil

end  # module

using .NilNumber: Nil, nil

"""
    outputsize(m, inputsize::Tuple; padbatch=false)

Calculate the output size of model `m` given the input size. 
Obeys `outputsize(m, size(x)) == size(m(x))` for valid input `x`.
Keyword `padbatch=true` is equivalent to using `(inputsize..., 1)`, and 
returns the final size including this extra batch dimension.

This should be faster than calling `size(m(x))`. It uses a trivial number type, 
and thus should work out of the box for custom layers.

If `m` is a `Tuple` or `Vector`, its elements are applied in sequence, like `Chain(m...)`.

# Examples
```jldoctest
julia> using Flux: outputsize

julia> outputsize(Dense(10, 4), (10,); padbatch=true)
(4, 1)

julia> m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32));

julia> m(randn(Float32, 10, 10, 3, 64)) |> size
(6, 6, 32, 64)

julia> outputsize(m, (10, 10, 3); padbatch=true)
(6, 6, 32, 1)

julia> outputsize(m, (10, 10, 3, 64))
(6, 6, 32, 64)

julia> try outputsize(m, (10, 10, 7, 64)) catch e println(e) end
DimensionMismatch("Input channels must match! (7 vs. 3)")

julia> outputsize([Dense(10, 4), Dense(4, 2)], (10, 1))
(2, 1)

julia> using LinearAlgebra: norm

julia> f(x) = x ./ norm.(eachcol(x));

julia> outputsize(f, (10, 1)) # manually specify batch size as 1
(10, 1)

julia> outputsize(f, (10,); padbatch=true) # no need to mention batch size
(10, 1)
```
"""
function outputsize(m, inputsize::Tuple; padbatch=false)
  inputsize = padbatch ? (inputsize..., 1) : inputsize
  
  return size(m(fill(nil, inputsize)))
end

## make tuples and vectors be like Chains

outputsize(m::Tuple, inputsize::Tuple; padbatch=false) = outputsize(Chain(m...), inputsize; padbatch=padbatch)
outputsize(m::AbstractVector, inputsize::Tuple; padbatch=false) = outputsize(Chain(m...), inputsize; padbatch=padbatch)

## bypass statistics in normalization layers

for layer in (:LayerNorm, :BatchNorm, :InstanceNorm, :GroupNorm)
  @eval (l::$layer)(x::AbstractArray{Nil}) = x
end

## fixes for layers that don't work out of the box

for (fn, Dims) in ((:conv, DenseConvDims), (:depthwiseconv, DepthwiseConvDims))
  @eval begin
    function NNlib.$fn(a::AbstractArray{Nil}, b::AbstractArray{Nil}, dims::$Dims)
      fill(nil, NNlib.output_size(dims)..., NNlib.channels_out(dims), size(a)[end])
    end

    function NNlib.$fn(a::AbstractArray{<:Real}, b::AbstractArray{Nil}, dims::$Dims)
      NNlib.$fn(fill(nil, size(a)), b, dims)
    end

    function NNlib.$fn(a::AbstractArray{Nil}, b::AbstractArray{<:Real}, dims::$Dims)
      NNlib.$fn(a, fill(nil, size(b)), dims)
    end
  end
end
