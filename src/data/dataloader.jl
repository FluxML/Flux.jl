# Adapted from Knet's src/data.jl (author: Deniz Yuret)

struct DataLoader{D}
    data::D
    batchsize::Int
    nobs::Int
    partial::Bool
    imax::Int
    indices::Vector{Int}
    shuffle
end

"""
    DataLoader(data; batchsize = 1, shuffle = x -> Random.shuffle(Random.GLOBAL_RNG, x), partial = true)

An object that iterates over mini-batches of `data`, each mini-batch containing `batchsize` observations
(except possibly the last one).

Takes as input a single data tensor, or a tuple (or a named tuple) of tensors.
The last dimension in each tensor is considered to be the observation dimension.

By default, the dataloader shuffles the observations each time iterations are re-started.
To not shuffle the data, pass `shuffle = identity`.

If `partial = false`, drops the last mini-batch if it is smaller than the batchsize.

The original data is preserved in the `data` field of the DataLoader.

Usage example:

    Xtrain = rand(10, 100)
    train_loader = DataLoader(Xtrain, batchsize=2)
    # iterate over 50 mini-batches of size 2
    for x in train_loader
        @assert size(x) == (10, 2)
        ...
    end

    train_loader.data   # original dataset

    # similar, but yielding tuples
    train_loader = DataLoader((Xtrain,), batchsize=2)
    for (x,) in train_loader
        @assert size(x) == (10, 2)
        ...
    end

    Xtrain = rand(10, 100)
    Ytrain = rand(100)
    train_loader = DataLoader((Xtrain, Ytrain), batchsize=2, shuffle=true)
    for epoch in 1:100
        for (x, y) in train_loader
            @assert size(x) == (10, 2)
            @assert size(y) == (2,)
            ...
        end
    end

    # train for 10 epochs
    using IterTools: ncycle
    Flux.train!(loss, ps, ncycle(train_loader, 10), opt)

    # can use NamedTuple to name tensors
    train_loader = DataLoader((images=Xtrain, labels=Ytrain), batchsize=2, shuffle=true)
    for datum in train_loader
        @assert size(datum.images) == (10, 2)
        @assert size(datum.labels) == (2,)
    end
"""
function DataLoader(data; batchsize = 1, shuffle = identity, partial = true)
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))

    n = _nobs(data)
    if n < batchsize
        @warn "Number of observations less than batchsize, decreasing the batchsize to $n"
        batchsize = n
    end
    imax = partial ? n : n - batchsize + 1
    DataLoader(data, batchsize, n, partial, imax, [1:n;], shuffle)
end

@propagate_inbounds function Base.iterate(d::DataLoader, i = 0)     # returns data in d.indices[i+1:i+batchsize]
    i >= d.imax && return nothing
    d.shuffle(d.indices)
    nexti = min(i + d.batchsize, d.nobs)
    ids = d.indices[i+1:nexti]
    batch = _getobs(d.data, ids)
    return (batch, nexti)
end

function Base.length(d::DataLoader)
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

_nobs(data::AbstractArray) = size(data)[end]

function _nobs(data::Union{Tuple, NamedTuple})
    length(data) > 0 || throw(ArgumentError("Need at least one data input"))
    n = _nobs(data[1])
    if !all(x -> _nobs(x) == n, Base.tail(data))
        throw(DimensionMismatch("All data should contain same number of observations"))
    end
    return n
end

_getobs(data::AbstractArray, i) = data[ntuple(i -> Colon(), Val(ndims(data) - 1))..., i]
_getobs(data::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(_getobs, i), data)

Base.eltype(::DataLoader{D}) where D = D
