using Random:randperm!

"""
    DataLoader(dataset::AbstractArray...; batchsize::Int, shuffle::Bool)

DataLoader provides iterators over the dataset.

```julia
X = rand(10, 1000)
Y = rand(1, 1000)

m = Dense(10, 1)
loss(x, y) = Flux.mse(m(x), y)
opt = ADAM(params(m))

trainloader = DataLoader(X, Y, batchsize=256, shuffle=true)

Flux.train!(loss, trainloader, opt)
```
"""
struct DataLoader
  dataset::Tuple
  batchsize::Int
  shuffle::Bool
  indices::Vector{Int}
  n::Int
end

function DataLoader(dataset::Tuple{AbstractArray, Vararg{AbstractArray}}; batchsize::Int, shuffle::Bool)
  l = last.(size.(dataset))
  n = first(l)
  all(n .== l) || throw(DimensionMismatch("All data should have the same length."))
  indices = collect(1:n)
  shuffle && randperm!(indices)
  DataLoader(dataset, batchsize, shuffle, indices, n)
end

DataLoader(dataset::AbstractArray...; batchsize::Int, shuffle::Bool) =
  DataLoader(dataset, batchsize=batchsize, shuffle=shuffle)

function Base.iterate(it::DataLoader, start=1)
  if start > it.n
      it.shuffle && randperm!(it.indices)
      return nothing
  end
  nextstart = min(start + it.batchsize, it.n + 1)
  i = it.indices[start:nextstart-1]
  element = Tuple(copy(selectdim(x, ndims(x), i)) for x in it.dataset)
  return element, nextstart
end

Base.length(it::DataLoader) = it.n
Base.eltype(it::DataLoader) = typeof(it.dataset)

function Base.show(io::IO, it::DataLoader)
  print(io, "DataLoader(dataset size = $(it.n)")
  print(io, ", batchsize = $(it.batchsize), shuffle = $(it.shuffle)")
  print(io, ")")
end
