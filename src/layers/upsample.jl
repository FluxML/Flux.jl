"""
    Upsample(; scale, mode)

An upsampling layer. `scale` is a tuple determining 
the output rescaling factor along each spatial dimension.
The only currently supported upsampling `mode` is 
`:bilinear`.

See [`NNlib.upsample_bilinear`](@ref).

# Examples

```juliarepl
julia> m = Upsample(scale=(2,3), mode=:bilinear)
Upsample{Val{:bilinear},2,Int64}((2, 3))

julia> m(ones(1,1,1,1))
2×3×1×1 Array{Float64,4}:
[:, :, 1, 1] =
 1.0  1.0  1.0
 1.0  1.0  1.0
```
"""
struct Upsample{Mode, N, T} 
  scale::NTuple{N, T}
end

function Upsample(; scale::NTuple{N, T}, mode::Symbol) where {N,T}
  @assert mode in [:bilinear] "Unsupported mode '$mode'"
  @assert N == 2  "Only 2d upsampling currently supported"
  Upsample{Val{mode}, N, T}(scale)
end

(m::Upsample{Val{:bilinear}})(x) = NNlib.upsample_bilinear(x, m.scale)


"""
    PixelShuffle(r::Int)

Pixel shuffling layer with upscale factor `r`.
 
See [`NNlib.pixel_shuffle`](@ref).
"""
struct PixelShuffle 
  r::Int
end

(m::PixelShuffle)(x) = NNlib.pixel_shuffle(x, m.r)
