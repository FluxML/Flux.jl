module NilNumber

using LinearAlgebra
using NNlib

"""
    Nil <: Number

Nil is a singleton type with a single instance `nil`.
Unlike `Nothing` and `Missing` it subtypes `Number`.
"""
struct Nil <: Number end

const nil = Nil()

Nil(::T) where T<:Number = nil
(::Type{T})(::Nil) where T<:Number = nil

Base.float(::Type{Nil}) = Nil
Base.copy(::Nil) = nil
Base.abs2(::Nil) = nil
Base.sqrt(::Nil) = nil
Base.zero(::Type{Nil}) = nil
Base.one(::Type{Nil}) = nil

for f in [copy, zero, one, oneunit, :+, :-, :abs, :abs2, :inv, :exp, :log]
    @eval Base.$f(::Nil) = nil
end

for f in [:+, :-, :*, :/, :mod, :div, :rem]
    @eval Base.$f(::Nil, ::Nil) = nil
end

Base.inv(::Nil) = nil

Base.isless(::Nil, ::Nil) = true
Base.isless(::Nil, ::Number) = true
Base.isless(::Number, ::Nil) = true

Base.isnan(::Nil) = false

Base.abs(::Nil) = nil
Base.exp(::Nil) = nil

Base.typemin(::Type{Nil}) = nil
Base.typemax(::Type{Nil}) = nil
Base.:^(::Nil, ::Nil) = nil

# TODO: can this be shortened?
Base.promote(x::Nil, y::Nil) = (nil, nil)
Base.promote(x::Nil, y) = (nil, nil)
Base.promote(x, y::Nil) = (nil, nil)
Base.promote(x::Nil, y, z) = (nil, nil, nil)
Base.promote(x, y::Nil, z) = (nil, nil, nil)
Base.promote(x, y, z::Nil) = (nil, nil, nil)
Base.promote(x::Nil, y, z::Nil) = (nil, nil, nil)
Base.promote(x::Nil, y::Nil, z::Nil) = (nil, nil, nil)
Base.promote(x::Nil, y::Nil, z) = (nil, nil, nil)


LinearAlgebra.adjoint(::Nil) = nil
LinearAlgebra.transpose(::Nil) = nil

end  # module

using .NilNumber: Nil, nil

"""
    _handle_batchin(isize, dimsize)

Gracefully handle ignoring batch dimension by padding `isize` with a 1 if necessary.
Also returns a boolean indicating if the batch dimension was padded.

# Arguments:
- `isize`: the input size as specified by the user
- `dimsize`: the expected number of dimensions for this layer (including batch)
"""
function _handle_batchin(isize, dimsize)
  indims = length(isize)
  @assert isnothing(dimsize) || indims == dimsize || indims == dimsize - 1
    "outdims expects ndims(isize) == $dimsize (got isize = $isize). isize should be the size of the input to the function (with batch size optionally left off)"
  
  return (indims == dimsize || isnothing(dimsize)) ? (isize, false) : ((isize..., 1), true)
end

"""
    _handle_batchout(outsize, ispadded)

Drop the batch dimension if requested.

# Arguments:
- `outsize`: the output size from a function
- `ispadded`: indicates whether the batch dimension in `outsize` is padded (see _handle_batchin)
"""
_handle_batchout(outsize, ispadded) = ispadded ? outsize[1:(end - 1)] : outsize

"""
    outdims(m, isize)

Calculate the output size of model/function `m` given an input of size `isize` (w/o computing results).
`isize` should include all dimensions (except batch dimension can be optionally excluded).
If `m` is a `Tuple` or `Vector`, `outdims` treats `m` like a `Chain`.

*Note*: this method should work out of the box for custom layers,
  but you may need to specify the batch size manually.
To take advantage of automatic batch dim handling for your layer, define [`dimhint`](@ref).

# Examples
```jldoctest
julia> outdims(Dense(10, 4), (10,))
(4,)

julia> m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32));

julia> m(randn(Float32, 10, 10, 3, 64)) |> size
(6, 6, 32, 64)

julia> outdims(m, (10, 10, 3))
(6, 6, 32)

julia> outdims(m, (10, 10, 3, 64))
(6, 6, 32, 64)

julia> try outdims(m, (10, 10, 7, 64)) catch e println(e) end
DimensionMismatch("Input channels must match! (7 vs. 3)")

julia> outdims([Dense(10, 4), Dense(4, 2)], (10,))
(2,)

julia> using LinearAlgebra: norm

julia> f(x) = x ./ norm.(eachcol(x));

julia> outdims(f, (10, 1)) # manually specify batch size as 1
(10, 1)

julia> Flux.dimhint(::typeof(f)) = 2; # our custom f expects 2D arrays (batch included)

julia> outdims(f, (10,)) # no need to mention batch size
(10,)
```
"""
function outdims(m, isize; preserve_batch = false)
  isize, ispadded = _handle_batchin(isize, dimhint(m))
  
  return _handle_batchout(size(m(fill(nil, isize))), ispadded)
end

## dimension hints

"""
    Flux.dimhint(m)

Return the expected dimensions of the input to a function.
So, for a function `f(x)`, `dimhint(f) == ndims(x)`.

Note that for [`Chain`](@ref), only the first layer must support
  `dimhint`.

Override this method for your custom layer to take advantage
  of the automatic batch handling in [`outdims`](@ref).
"""
dimhint(m) = nothing
dimhint(m::Tuple) = dimhint(first(m))
dimhint(m::Chain) = dimhint(m.layers)
dimhint(::Dense) = 2
dimhint(::Diagonal) = 2
dimhint(m::Maxout) = dimhint(first(m.over))
dimhint(m::SkipConnection) = dimhint(m.layers)
dimhint(m::Conv) = ndims(m.weight)
dimhint(::ConvTranspose) = 4
dimhint(::DepthwiseConv) = 4
dimhint(::CrossCor) = 4
dimhint(::MaxPool) = 4
dimhint(::MeanPool) = 4
dimhint(::AdaptiveMaxPool) = 4
dimhint(::AdaptiveMeanPool) = 4
dimhint(::GlobalMaxPool) = 4
dimhint(::GlobalMeanPool) = 4

## make tuples and vectors be like Chains

outdims(m::Tuple, isize) = outdims(Chain(m...), isize)
outdims(m::AbstractVector, isize) = outdims(Chain(m...), isize)

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
