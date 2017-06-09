export reshape

import Base: reshape

reshape(x::AbstractArray, dims::AbstractArray) = reshape(x,tuple(dims...))
