"""
  Upsample(scale; mode=:nearest)  
  Upsample(; size, mode=:bilinear)

An upsampling layer. `scale` is a number or a tuple of numbers 
representing  the output rescaling factor along each spatial dimension.

Currently supported upsampling `mode`s are:
  - `:nearest`
  - `:bilinear`

For some `mode`s it is possible
to directly specify the output spatial `size`,
as an alternative to `scale`,   

See [`NNlib.upsample_nearest`](@ref), [`NNlib.upsample_bilinear`](@ref).

# Examples

```juliarepl
julia> m = Upsample((2,3), mode=:bilinear)
Upsample((2, 3), mode=bilinear)

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

function Upsample(scale; mode::Symbol=:nearest)
  mode in [:nearest, :bilinear] || 
    throw(ArgumentError("mode=:$mode is not supported."))
  return Upsample(scale, nothing)
end

function Upsample(; size, mode::Symbol=:bilinear)
  mode in [:nearest, :bilinear] || 
    throw(ArgumentError("mode=:$mode is not supported."))
  return Upsample(nothing, size)
end

(m::Upsample{:nearest})(x::AbstractArray) =
  NNlib.upsample_nearest(x, m.scale)

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


