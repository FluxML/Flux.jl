# Custom Dataset

In order to maintain compatibility for custom datasets with `DataLoader`
you need to implement following methods:

- `Flux.Data.nobs(::CustomDataset)` -- total number of items in `CustomDataset`;
- `Flux.Data.getobs(::CustomDataset, ids)` -- how to retrieve items from dataset for given list of `ids`;
- `Base.eltype(::DataLoader{CustomDataset})` -- type of the elements returned by dataset.

Below is a dummy example of how to adapt custom dataset
to make it compatible with `DataLoader`.

```julia
# For each index returns an array of zeros.
struct CustomDataset{T, N}
    element_size::Tuple
    total::Int
end

Base.eltype(::DataLoader{CustomDataset{T, N}}) where {T, N} = Array{T, N}

Flux.Data.nobs(d::CustomDataset) = d.total
function Flux.Data.getobs(d::CustomDataset{T, N}, i)::Array{T, N} where {T, N}
    zeros(T, d.element_size..., length(i))
end
```

And now you can use `CustomDataset` with `DataLoader`:

```julia
dataset = CustomDataset{Float32, 4}((28, 28, 1), 16)
loader = DataLoader(dataset, batchsize=4, shuffle=true)
batches = collect(loader)
```
