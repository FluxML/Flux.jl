# Adapted from Knet's src/data.jl (author: Deniz Yuret)
using Random: AbstractRNG, shuffle!, GLOBAL_RNG

struct DataLoader{D,R<:AbstractRNG}
    data::D
    batchsize::Int
    nobs::Int
    partial::Bool
    imax::Int
    indices::Vector{Int}
    shuffle::Bool
    rng::R
end

"""
    DataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)

An object that iterates over mini-batches of `data`, 
each mini-batch containing `batchsize` observations
(except possibly the last one).

Takes as input a single `data` array, a tuple/namedtuple/dictionary of arrays,
or more generally any type implementing the [`LearnBase.nobs`](@ref) 
and [`LearnBase.getobs`](@ref) interfaces.
The last dimension in each array is the observation dimension, i.e. the one
divided into mini-batches.

If `shuffle=true`, it shuffles the observations each time iterations are re-started.
If `partial=false` and the number of observations is not divisible by the batchsize, 
then the last mini-batch is dropped.

The original data is preserved in the `data` field of the DataLoader.

# Examples
```jldoctest
julia> Xtrain = rand(10, 100);

julia> array_loader = Flux.DataLoader(Xtrain, batchsize=2);

julia> for x in array_loader
         @assert size(x) == (10, 2)
         # do something with x, 50 times
       end

julia> array_loader.data === Xtrain
true

julia> tuple_loader = Flux.DataLoader((Xtrain,), batchsize=2);  # similar, but yielding 1-element tuples

julia> for x in tuple_loader
         @assert x isa Tuple{Matrix}
         @assert size(x[1]) == (10, 2)
       end

julia> Ytrain = rand('a':'z', 100);  # now make a DataLoader yielding 2-element named tuples

julia> train_loader = Flux.DataLoader((data=Xtrain, label=Ytrain), batchsize=5, shuffle=true);

julia> for epoch in 1:100
         for (x, y) in train_loader  # access via tuple destructuring
           @assert size(x) == (10, 5)
           @assert size(y) == (5,)
           # loss += f(x, y) # etc, runs 100 * 20 times
         end
       end

julia> first(train_loader).label isa Vector{Char}  # access via property name
true

julia> first(train_loader).label == Ytrain[1:5]  # because of shuffle=true
false

julia> foreach(println∘summary, Flux.DataLoader(rand(Int8, 10, 64), batchsize=30))  # partial=false would omit last
10×30 Matrix{Int8}
10×30 Matrix{Int8}
10×4 Matrix{Int8}
```
"""
function DataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))

    n = nobs(data)
    if n < batchsize
        @warn "Number of observations less than batchsize, decreasing the batchsize to $n"
        batchsize = n
    end
    imax = partial ? n : n - batchsize + 1
    DataLoader(data, batchsize, n, partial, imax, [1:n;], shuffle, rng)
end

@propagate_inbounds function Base.iterate(d::DataLoader, i=0)     # returns data in d.indices[i+1:i+batchsize]
    i >= d.imax && return nothing
    if d.shuffle && i == 0
        shuffle!(d.rng, d.indices)
    end
    nexti = min(i + d.batchsize, d.nobs)
    ids = d.indices[i+1:nexti]
    batch = getobs(d.data, ids)
    return (batch, nexti)
end

function Base.length(d::DataLoader)
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end
