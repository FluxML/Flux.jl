export reshape, tile, fill, cast, solve, triangular_solve, randu, randn,
       expand_dims

import Base: reshape, fill, randn

reshape(x::AbstractArray, dims::AbstractArray) = reshape(x,tuple(dims...))
tile(x::AbstractArray, mult::AbstractArray) = repeat(x,outer=tuple(mult...))
fill{T}(x::T, dims::AbstractArray) = fill(x,tuple(dims...))
cast{T}(x::AbstractArray, ::Type{T}) = convert(Array{T},x)
solve(A::AbstractArray, b::AbstractArray) = A\b
triangular_solve(A::AbstractArray, b::AbstractArray) = A\b
randu(x::AbstractArray) = rand(tuple(x...))
randn(x::AbstractArray) = randn(tuple(x...))

function expand_dims(x,dim)
    s = [size(x)...]
    reshape(x,tuple(vcat(s[1:dim-1],1,s[dim:end])...))
end
