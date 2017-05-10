export tile

tile(x::AbstractArray, mult::AbstractArray) = repeat(x,outer=tuple(mult...))
