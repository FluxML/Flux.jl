"""
    Upsample(mode = :nearest; [scale, size]) 
    Upsample(scale, mode = :nearest)  

An upsampling layer. One of two keywords must be given:

If `scale` is a number, this applies to all but the last two dimensions (channel and batch) of the input. 
It may also be a tuple, to control dimensions individually. Alternatively, keyword 
`size` accepts a tuple, to directly specify the leading dimensions of the output.

Currently supported upsampling `mode`s 
and corresponding NNlib's methods are:
  - `:nearest` -> [`NNlib.upsample_nearest`](@ref) 
  - `:bilinear` -> [`NNlib.upsample_bilinear`](@ref)
  - `:trilinear` -> [`NNlib.upsample_trilinear`](@ref)

# Examples

```jldoctest
julia> m = Upsample(scale = (2, 3))
Upsample(:nearest, scale = (2, 3))

julia> m(ones(2, 2, 1, 1)) |> size
(4, 6, 1, 1)

julia> m = Upsample(:bilinear, size = (4, 5))
Upsample(:bilinear, size = (4, 5))

julia> m(ones(2, 2, 1, 1)) |> size
(4, 5, 1, 1)
```
"""
struct Upsample{mode, S, T}
  scale::S
  size::T
end

Functors.@leaf Upsample # mark leaf since the constructor is not compatible with Functors
                        # by default but we don't need to recurse into it   

function Upsample(mode::Symbol = :nearest; scale = nothing, size = nothing)
  mode in [:nearest, :bilinear, :trilinear] || 
    throw(ArgumentError("mode=:$mode is not supported."))
  if !(isnothing(scale) ⊻ isnothing(size))
    throw(ArgumentError("Either scale or size should be specified (but not both)."))
  end
  return Upsample{mode,typeof(scale),typeof(size)}(scale, size)
end

Upsample(scale, mode::Symbol = :nearest) = Upsample(mode; scale)

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

(m::Upsample{:trilinear})(x::AbstractArray) =
  NNlib.upsample_trilinear(x, m.scale)
(m::Upsample{:trilinear, Nothing})(x::AbstractArray) = 
  NNlib.upsample_trilinear(x; size=m.size)

function Base.show(io::IO, u::Upsample{mode}) where {mode}
  print(io, "Upsample(")
  print(io, ":", mode)
  u.scale !== nothing && print(io, ", scale = $(u.scale)")
  u.size !== nothing && print(io, ", size = $(u.size)")
  print(io, ")")
end

"""
    PixelShuffle(r::Int)

Pixel shuffling layer with upscale factor `r`. Usually used for generating higher
resolution images while upscaling them.
 
See [`NNlib.pixel_shuffle`](@ref).

# Examples
```jldoctest
julia> p = PixelShuffle(2);

julia> xs = [2row + col + channel/10 for row in 1:2, col in 1:2, channel in 1:4, n in 1:1]
2×2×4×1 Array{Float64, 4}:
[:, :, 1, 1] =
 3.1  4.1
 5.1  6.1

[:, :, 2, 1] =
 3.2  4.2
 5.2  6.2

[:, :, 3, 1] =
 3.3  4.3
 5.3  6.3

[:, :, 4, 1] =
 3.4  4.4
 5.4  6.4

julia> p(xs)
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 3.1  3.3  4.1  4.3
 3.2  3.4  4.2  4.4
 5.1  5.3  6.1  6.3
 5.2  5.4  6.2  6.4

julia> xs = [3row + col + channel/10 for row in 1:2, col in 1:3, channel in 1:4, n in 1:1]
2×3×4×1 Array{Float64, 4}:
[:, :, 1, 1] =
 4.1  5.1  6.1
 7.1  8.1  9.1

[:, :, 2, 1] =
 4.2  5.2  6.2
 7.2  8.2  9.2

[:, :, 3, 1] =
 4.3  5.3  6.3
 7.3  8.3  9.3

[:, :, 4, 1] =
 4.4  5.4  6.4
 7.4  8.4  9.4

julia> p(xs)
4×6×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 4.1  4.3  5.1  5.3  6.1  6.3
 4.2  4.4  5.2  5.4  6.2  6.4
 7.1  7.3  8.1  8.3  9.1  9.3
 7.2  7.4  8.2  8.4  9.2  9.4
```
"""
struct PixelShuffle 
  r::Int
end

(m::PixelShuffle)(x) = NNlib.pixel_shuffle(x, m.r)
