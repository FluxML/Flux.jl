"""
  Upsample(scale; mode=:nearest)  
  Upsample(; size, mode=:nearest)

An upsampling layer. 

`scale` is a number or a tuple of numbers 
representing  the output rescaling factor along each spatial dimension.
For integer `scale`, all but the last 2 dimensions (channel and batch)
will rescaled by the same factor. 

It is also possible to directly specify the output spatial `size`,
as an alternative to using `scale`.

Currently supported upsampling `mode`s 
and corresponding NNlib's methods are:
  - `:nearest` -> [`NNlib.upsample_nearest`](@ref) 
  - `:bilinear` -> [`NNlib.upsample_bilinear`](@ref)

# Examples

```juliarepl
julia> m = Upsample((2,3), mode=:bilinear)
Upsample((2, 3), mode=:bilinear)

julia> m(ones(1,1,1,1))
2×3×1×1 Array{Float64,4}:
[:, :, 1, 1] =
 1.0  1.0  1.0
 1.0  1.0  1.0
```
"""
struct Upsample{Mode,S,T}
  scale::S
  size::T
end

function Upsample(scale::S; mode::Symbol=:nearest) where S
  mode in [:nearest, :bilinear] || 
    throw(ArgumentError("mode=:$mode is not supported."))
  return Upsample{mode,S,Nothing}(scale, nothing)
end

function Upsample(; size::T, mode::Symbol=:nearest) where T
  mode in [:nearest, :bilinear] || 
    throw(ArgumentError("mode=:$mode is not supported."))
  return Upsample{mode, Nothing, T}(nothing, size)
end

(m::Upsample{:nearest})(x::AbstractArray) =
  NNlib.upsample_nearest(x, m.scale)
function (m::Upsample{:nearest, Int})(x::AbstractArray{T, N}) where {T, N} 
  NNlib.upsample_nearest(x, ntuple(i -> m.scale, N-2))
end
(m::Upsample{:nearest, Nothing})(x::AbstractArray) =
  NNlib.upsample_nearest(x; size=m.size)

(m::Upsample{:bilinear})(x::AbstractArray) =
  NNlib.upsample_bilinear(x, m.scale)
(m::Upsample{:bilinear, Nothing})(x::AbstractArray) = 
  NNlib.upsample_bilinear(x; size=m.size)

function Base.show(io::IO, u::Upsample{mode}) where {mode}
  print(io, "Upsample(")
  u.scale !== nothing && print(io, "$(u.scale), ")
  u.size !== nothing && print(io, "size=$(u.size), ")
  print(io, "mode=:", mode)
  println(io, ")")
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


