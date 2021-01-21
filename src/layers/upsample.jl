"""
  Upsample(scale; mode=:nearest)  
  Upsample(; scale, mode=:nearest)

An upsampling layer. `scale` is a number or a tuple of numbers 
representing  the output rescaling factor along each spatial dimension.
Currently supported upsampling `mode`s are:
  - `:nearest`
  - `:bilinear`

See [`NNlib.upsample_nearest`](@ref), [`NNlib.upsample_bilinear`](@ref).

# Examples

```juliarepl
julia> m = Upsample(scale=(2,3), mode=:bilinear)
Upsample(scale=(2, 3), mode=bilinear)

julia> m(ones(1,1,1,1))
2×3×1×1 Array{Float64,4}:
[:, :, 1, 1] =
 1.0  1.0  1.0
 1.0  1.0  1.0
```
"""
struct Upsample{Mode,T}
  scale::T
end

Upsample(; scale, mode::Symbol=:nearest) = Upsample(scale; mode)

function Upsample(scale; mode::Symbol=:nearest)
  mode in [:nearest, :bilinear] || 
    throw(ArgumentError("mode=:$mode is not supported, allowed values are :nearest, :bilinear"))
  return Upsample{mode, typeof(scale)}(scale)
end

(m::Upsample{:nearest,<:Number})(x::AbstractArray{T,N}) where {T,N} =
  NNlib.upsample_nearest(x, ntuple(_ -> m.scale, N-2))
(m::Upsample{:nearest,<:Tuple})(x::AbstractArray) =
  NNlib.upsample_nearest(x, m.scale)

(m::Upsample{:bilinear,<:Number})(x::AbstractArray{T,N}) where {T,N} =
  NNlib.upsample_bilinear(x, ntuple(_ -> m.scale, N-2))
(m::Upsample{:bilinear,<:Tuple})(x::AbstractArray) =
  NNlib.upsample_bilinear(x, m.scale)

function Base.show(io::IO, u::Upsample{mode}) where {mode}
  print(io, "Upsample(scale=", u.scale)
  print(io, ", mode=:", mode)
  print(io, ")")
end


"""
    PixelShuffle(r::Int)

Pixel shuffling layer with upscale factor `r`.
 
See [`NNlib.pixel_shuffle`](@ref).
"""
struct PixelShuffle 
  r::Int
end

(m::PixelShuffle)(x) = NNlib.pixel_shuffle(x, m.r)
