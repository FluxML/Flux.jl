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

# Examples

```juliarepl
julia> m = Upsample(scale = (2, 3))
Upsample(:nearest, scale = (2, 3))

julia> m(ones(2, 2, 1, 1)) |> size
(4, 6, 1, 1)

julia> m = Upsample(:bilinear, size = (4, 5))
Upsample(:bilinear, size = (4, 5))

julia> m(ones(2, 2, 1, 1)) |> size
(4, 5, 1, 1)
"""
struct Upsample{mode, S, T}
  scale::S
  size::T
end

function Upsample(mode::Symbol = :nearest; scale = nothing, size = nothing)
  mode in [:nearest, :bilinear] || 
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

function Base.show(io::IO, u::Upsample{mode}) where {mode}
  print(io, "Upsample(")
  print(io, ":", mode)
  u.scale !== nothing && print(io, ", scale = $(u.scale)")
  u.size !== nothing && print(io, ", size = $(u.size)")
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
