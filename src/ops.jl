export reshape, tile, fill, slice, pad, cast, randu, randn, solve, triangular_solve,
       expand_dims, gather

import Base: reshape, fill, randn

reshape(x::AbstractArray, dims::AbstractArray) = reshape(x,tuple(dims...))
tile(x::AbstractArray, mult::AbstractArray) = repeat(x,outer=tuple(mult...))
fill{T}(x::T, dims::AbstractArray) = fill(x,tuple(dims...))
cast{T}(x::AbstractArray, ::Type{T}) = convert(Array{T},x)
randu(x::AbstractArray) = rand(tuple(x...))
randn(x::AbstractArray) = randn(tuple(x...))
solve(A::AbstractArray, b::AbstractArray) = A\b
triangular_solve(A::AbstractArray, b::AbstractArray) = A\b

function slice(x::AbstractArray, be::AbstractArray, si::AbstractArray)
    s = size(x)
    ndims = length(s)
    @assert length(be) == ndims
    @assert length(si) == ndims
    inds = Vector{UnitRange{Int}}(ndims)
    for i in 1:ndims
        inds[i] = si[i] == -1 ? range(be[i],s[i]-be[i]+1) : range(be[i],si[i])
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

function expand_dims(x,dim)
    s = [size(x)...]
    reshape(x,tuple(vcat(s[1:dim-1],1,s[dim:end])...))
end

gather(x::AbstractArray,ind::Int) = gather(x,[ind])[:]
function gather(x::AbstractArray,inds::AbstractArray)
    s = convert(Array{Any},[size(x)...])
    si = convert(Array{Any},[size(inds)...])
    ret_dims = [si;s[2:end]]
    ret = reshape(resize!(copy(x[1:1])[:],prod(ret_dims)),ret_dims)
    nd = length(s)
    for i in 2:nd
        s[i] = Colon()
    end
    ndi = length(si)
    ret_dims[ndi+1:end] = Colon()
    for i in 1:length(inds)
        ret_dims[1:ndi] = [ind2sub(inds,i)...]
        s[1] = inds[i]
        ret[ret_dims...] = x[s...]
    end
    ret
end
