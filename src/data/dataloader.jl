# # Adapted from Knet's src/data.jl (author: Deniz Yuret)
# 
# struct DataLoader{F, D}
#     iterator::F
#     data::D
#     batchsize::Int
#     nobs::Int
#     partial::Bool
#     imax::Int
#     indices::Vector{Int}
#     shuffle::Bool
# end
# 
# """
#     DataLoader(data; batchsize=1, shuffle=false, partial=true)
# 
# An object that iterates over mini-batches of `data`, each mini-batch containing `batchsize` observations
# (except possibly the last one).
# 
# Takes as input a single data tensor, or a tuple (or a named tuple) of tensors.
# The last dimension in each tensor is considered to be the observation dimension.
# 
# If `shuffle=true`, shuffles the observations each time iterations are re-started.
# If `partial=false`, drops the last mini-batch if it is smaller than the batchsize.
# 
# The original data is preserved in the `data` field of the DataLoader.
# 
# Usage example:
# 
#     Xtrain = rand(10, 100)
#     train_loader = DataLoader(Xtrain, batchsize=2)
#     # iterate over 50 mini-batches of size 2
#     for x in train_loader
#         @assert size(x) == (10, 2)
#         ...
#     end
# 
#     train_loader.data   # original dataset
# 
#     # similar, but yielding tuples
#     train_loader = DataLoader((Xtrain,), batchsize=2)
#     for (x,) in train_loader
#         @assert size(x) == (10, 2)
#         ...
#     end
# 
#     Xtrain = rand(10, 100)
#     Ytrain = rand(100)
#     train_loader = DataLoader((Xtrain, Ytrain), batchsize=2, shuffle=true)
#     for epoch in 1:100
#         for (x, y) in train_loader
#             @assert size(x) == (10, 2)
#             @assert size(y) == (2,)
#             ...
#         end
#     end
# 
#     # train for 10 epochs
#     using IterTools: ncycle
#     Flux.train!(loss, ps, ncycle(train_loader, 10), opt)
# 
#     # can use NamedTuple to name tensors
#     train_loader = DataLoader((images=Xtrain, labels=Ytrain), batchsize=2, shuffle=true)
#     for datum in train_loader
#         @assert size(datum.images) == (10, 2)
#         @assert size(datum.labels) == (2,)
#     end
# """
# function DataLoader(data; batchsize=1, shuffle=false, partial=true, f = dataloader_iterate)
#     batchsize > 0 || throw(ArgumentError("Need positive batchsize"))
# 
#     n = _nobs(data)
#     if n < batchsize
#         @warn "Number of observations less than batchsize, decreasing the batchsize to $n"
#         batchsize = n
#     end
#     imax = partial ? n : n - batchsize + 1
#     DataLoader(f, data, batchsize, n, partial, imax, [1:n;], shuffle)
# end
# 
# function getobs(d, i, args...)
#   i >= d.imax && return nothing
#     if d.shuffle && i == 0
#         shuffle!(d.indices)
#     end
#     nexti = min(i + d.batchsize, d.nobs)
#     ids = d.indices[i+1:nexti]
#     batch = _getobs(d.data, ids)
#     return (batch, nexti)
# end
# 
# @propagate_inbounds function Base.iterate(d::DataLoader, i = 0)     # returns data in d.indices[i+1:i+batchsize]
#   d.iterator(d, i)
# end
# 
# function Base.length(d::DataLoader)
#     n = d.nobs / d.batchsize
#     d.partial ? ceil(Int,n) : floor(Int,n)
# end
# 
# _nobs(data::AbstractArray) = size(data)[end]
# 
# function _nobs(data::Union{Tuple, NamedTuple})
#     length(data) > 0 || throw(ArgumentError("Need at least one data input"))
#     n = _nobs(data[1])
#     if !all(x -> _nobs(x) == n, Base.tail(data))
#         throw(DimensionMismatch("All data should contain same number of observations"))
#     end
#     return n
# end
# 
# _getobs(data::AbstractArray, i) = data[ntuple(i -> Colon(), Val(ndims(data) - 1))..., i]
# _getobs(data::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(_getobs, i), data)
# 
# Base.eltype(::DataLoader{F, D}) where {F,D} = D
# 

using Random: shuffle

struct DataLoader{F,S,D,L}
  aug::F
  data::D
  # args::D
  # labels::L
  batchsize::Int
  batchdim::Int
end

getobs(data::AbstractArray, n, i) = data
getobs(x, n, i) = getobs(Base.tail(x), n, i)

# X :: tuple of args to loss
function DataLoader(f
                    args...;
                    batchsize = 1, shuffle = true,
                    partial = true, batchdim = nothing)
  DataLoader(f, shuffle(args), batchsize, batchdim)
end

# `f` is an augmentation on the data/ labels
function DataLoader(f,
                    args::Vararg{AbstractArray};
                    batchsize = 1, shuffle = identity,
                    partial = true, batchdim = ndims)

  feats = first(args)
  ix = shuffle ? shuffle(1:size(feats, batchdim(feats))) : 1:size(feats, batchdim(feats))
  fs = foreach(feat -> getindex(feat, ntuple(Colon(), size(feat) - 1)..., ix), args)
  DataLoader(f, fs, batchsize, batchdim(feats))  
end

function DataLoader(args::Vararg{AbstractArray};
                    batchsize = 1, shuffle = identity,
                    partial = true, batchdim = ndims)

  DataLoader(identity, args...,
             batchsize = batchsize,
             shuffle = shuffle,
             partial = partial,
             batchdim = batchdim)
end

Base.length(dl::DataLoader) = size(first(dl.data)) ÷ dl.batchsize

# (dl::DataLoader){typeof(getobs)}(i) = getobs(dl, i)

function getobs(data::NTuple{N,AbstractArray}, ix, bd) where N
  foreach(d -> getindex(d, ntuple(i -> i == bd ? ix : Colon(), ndims(d)),), data)
end

function Base.iterate(dl::DataLoader, i = 0)
  Base.iterate.(dl.data, i)
end

function Base.iterate(dl::DataLoader{f, <:NTuple{N, AbstractArray}}, i) where {N,f}
  total = 
  ix = 
  args = getobs(dl.data, ix, bd)
  dl.aug.(args), i + 1
end
