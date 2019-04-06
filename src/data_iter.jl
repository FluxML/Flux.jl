using Base.Iterators: partition
using Random:shuffle

"""
Implemntation of iterators for batches.
Given an AbstractArray of datapoints, `batch_iter` will return an iterator over
batches of data of size `batch_size`.

Usage : 
```
julia> x = rand(28,28,3,101)
	   28×28×3×7 Array{Float64,4}: ...

julia> bi = batch_iter(x,3) # An iterator over batches of data

julia> first(bi)
	   28×28×3×3 Array{Float64,4}: ...
```
"""

mutable struct BatchIter{T,N}
    c::AbstractArray{T,N} # Data Array
    n::Int # Size of batch
    ind::Any # indices
end

function _batch_iter(c::AbstractArray,n::Int,ind::Any)
    b = BatchIter(c,n,ind)
    indices = collect(1:size(c)[end])[shuffle(1:end)]
    ind = collect(partition(indices,n))
    b.ind = ind
    return b
end

batch_iter(c::AbstractArray,n::Int) = _batch_iter(c,n,nothing)

function Base.iterate(b::BatchIter,state = 1)
    state <= length(ind) || return nothing
    return (b.c[[Colon() for _ in 1:ndims(b.c)-1]...,b.ind[state]],state+1)
end
