"""
  Upsample(mode = :nearest; scale = nothing, size = nothing)  

An upsampling layer. 

`scale` is a number or a tuple of numbers 
representing  the output rescaling factor along each spatial dimension.
For integer `scale`, all but the last 2 dimensions (channel and batch)
will be rescaled by the same factor. 

It is also possible to directly specify the output spatial `size`,
as an alternative to using `scale`.

Currently supported upsampling `mode`s 
and corresponding NNlib's methods are:
  - `:nearest` -> [`NNlib.upsample_nearest`](@ref) 
  - `:bilinear` -> [`NNlib.upsample_bilinear`](@ref)

# Examples

```juliarepl
julia> m = Upsample(scale = (2, 3))
Upsample(:nearest, scale=(2, 3))

julia> m(ones(2, 2, 1, 1)) |> size
(4, 6, 1, 1)

julia> m = Upsample(:bilinear, size = (4, 5))
Upsample(:bilinear, size=(4, 5))

julia> m(ones(2, 2, 1, 1)) |> size
(4, 5, 1, 1)
"""
struct Upsample{Mode,S,T}
  scale::S
  size::T
end

function Upsample(mode = :nearest; scale = nothing, size = nothing)
  mode in [:nearest, :bilinear] || 
    throw(ArgumentError("mode=:$mode is not supported."))
  if ~((scale === nothing) ⊻ (size === nothing))
    throw(ArgumentError("Either scale or size should be specified."))
  end
  return Upsample{mode,typeof(scale),typeof(size)}(scale, size)
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

