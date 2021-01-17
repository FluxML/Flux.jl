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
struct Upsample{Mode,T}
  scale::T
end

function Upsample(; scale, mode::Symbol=:bilinear)
  mode in [:bilinear] || throw(ArgumentError("only bilinear sampling is supported"))
  scale isa Integer || scale isa NTuple{2,<:Integer} || 
    throw(ArgumentError("only two-dimensional scaling by an integer factor is supported"))
  Upsample{mode, typeof(scale)}(scale)
end

(m::Upsample{:bilinear})(x) = NNlib.upsample_bilinear(x, m.scale isa Tuple ? m.scale : (m.scale, m.scale))

function Base.show(io::IO, u::Upsample2{mode}) where {mode}
  print(io, "Upsample(scale=", u.scale)
  mode == :bilinear || print(io, ", mode=", mode)
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
