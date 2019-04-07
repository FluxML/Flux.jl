using Base.Iterators: partition
using Random:shuffle

"""
Implemntation of iterators for batches.
Given an AbstractArray of datapoints, `batch_iter` will return an iterator over
batches of data of size `batch_size`.

Usage : 
```
julia> x = rand(28,28,3,7)
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

"""
Implemntation of iterators for epochs.
Given a BatchIterator, `epoch_iter` will return an iterator over all batches 
for the given number of epochs `ne`.

Usage: 
```
julia> x = rand(28,28,3,7)
       28×28×3×7 Array{Float64,4}: ...

julia> bi = batch_iter(x,3) # An iterator over batches of data 

julia> ei = epoch_iter(bi,3) # Iterator over batches for 3 epochs

julia> first(ei)
       28×28×3×3 Array{Float64,4}: ...
```
"""

mutable struct EpochIter{T,N}
    b::BatchIter{T,N}
    ne::Int # Number of epochs
end

epoch_iter(b::BatchIter,ne::Int) = EpochIter(b,ne)

function Base.iterate(e::EpochIter,state = 1)
    state <= length(e.b.ind)*e.ne || return nothing
    s = mod(state,length(e.b.ind))
    if s == 0
        s = length(e.b.ind)
    end
    return (e.b.c[[Colon() for _ in 1:ndims(e.b.c)-1]...,e.b.ind[s]],state+1)
end

function Base.length(e::EpochIter)
    l = size(e.b.c)[end]
    return (div(l, e.b.n) + ((mod(l, e.b.n) > 0) ? 1 : 0))*e.ne
end