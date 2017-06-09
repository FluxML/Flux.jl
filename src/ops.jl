export reshape, tile

import Base: reshape

reshape(x::AbstractArray, dims::AbstractArray) = reshape(x,tuple(dims...))
tile(x::AbstractArray, mult::AbstractArray) = repeat(x,outer=tuple(mult...))
