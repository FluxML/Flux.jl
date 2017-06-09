export reshape, tile, fill, cast

import Base: reshape, fill

reshape(x::AbstractArray, dims::AbstractArray) = reshape(x,tuple(dims...))
tile(x::AbstractArray, mult::AbstractArray) = repeat(x,outer=tuple(mult...))
fill{T}(x::T, dims::AbstractArray) = fill(x,tuple(dims...))
cast{T}(x::AbstractArray, ::Type{T}) = convert(Array{T},x)
