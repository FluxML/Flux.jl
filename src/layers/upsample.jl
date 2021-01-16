"""
    Upsample(; scale_factor, mode)

An upsampling layer.
`scale_factor`: a tuple of ints. 
`mode`: only `:bilinear` currently supported. 

See [`NNlib.upsample_bilinear`](@ref).
"""
struct Upsample{Mode, N, T} 
  scale_factor::NTuple{N, T}
end

function Upsample(; scale_factor::NTuple{N, T}, mode::Symbol) where {N,T}
  @assert mode in [:bilinear] "Not supported mode '$mode'"
  @assert N == 2  "Only 2d upsampling currently supported"
  Upsample{Val{mode}, N, T}(scale_factor)
end

(m::Upsample{Val{:bilinear}})(x) = NNlib.upsample_bilinear(x, m.scale_factor)


"""
    PixelShuffle(r::Int)

Pixel shuffling layer with upscale factor `r`.
 
See [`NNlib.pixel_shuffle`](@ref).
"""
struct PixelShuffle 
  r::Int
end

(m::PixelShuffle)(x) = NNlib.pixel_shuffle(x, m.r)
