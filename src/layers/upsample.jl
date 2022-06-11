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
```jldoctest; filter = r"[+-]?([0-9]*[.])?[0-9]+"
julia> p = PixelShuffle(2);

julia> xs = rand(2, 2, 4, 1)  # an image with 4 channels having 2X2 pixels in each channel
2×2×4×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.826452   0.0519244
 0.0686387  0.438346

[:, :, 2, 1] =
 0.343179  0.445101
 0.543927  0.740905

[:, :, 3, 1] =
 0.105997  0.422996
 0.32957   0.167205

[:, :, 4, 1] =
 0.825737  0.98609
 0.757365  0.294784

julia> p(xs)  # upsampled image with only 1 channel
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.826452   0.105997  0.0519244  0.422996
 0.343179   0.825737  0.445101   0.98609
 0.0686387  0.32957   0.438346   0.167205
 0.543927   0.757365  0.740905   0.294784
```
"""
struct PixelShuffle 
  r::Int
end

(m::PixelShuffle)(x) = NNlib.pixel_shuffle(x, m.r)
