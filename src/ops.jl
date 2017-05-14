export tile, fill, slice, pad, cast

import Base: fill

tile(x::AbstractArray, mult::AbstractArray) = repeat(x,outer=tuple(mult...))
fill{T}(x::T, dims::AbstractArray) = fill(x,tuple(dims...))
cast{T}(x::AbstractArray, tp::Type{T}) = convert(Array{T},x)

function slice(x::AbstractArray, be::AbstractArray, si::AbstractArray)
    s = size(x)
    ndims = length(s)
    @assert length(be) == ndims
    @assert length(si) == ndims
    inds = Vector{UnitRange{Int}}(ndims)
    for i in 1:ndims
        inds[i] = si[i] == -1 ? range(be[i],s[i]-be[i]) : range(be[i],si[i])
    end
    x[inds...]
end

function pad(x::AbstractArray, paddings::AbstractArray)
    s = size(x)
    ndims = length(s)
    @assert size(paddings) == (ndims,2)
    z = typeof(x[1,1])(0)
    ret = x
    for i in 1:ndims
        tmp = [size(ret)...]
        tmp[i] = paddings[i,1]
        ret = cat(i,fill(z,tmp),ret)
        tmp[i] = paddings[i,2]
        ret = cat(i,ret,fill(z,tmp))
    end
    ret
end
