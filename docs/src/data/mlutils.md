# Working with Data, using MLUtils.jl

Flux re-exports the `DataLoader` type and utility functions for working with
data from [MLUtils](https://github.com/JuliaML/MLUtils.jl).

## `DataLoader`

The `DataLoader` can be used to create mini-batches of data, in the format [`train!`](@ref) expects.

`Flux`'s website has a [dedicated tutorial](https://fluxml.ai/tutorials/2021/01/21/data-loader.html) on `DataLoader` for more information. 

```@docs
MLUtils.DataLoader
```

## Utility Functions

The utility functions are meant to be used while working with data;
these functions help create inputs for your models or batch your dataset.

```@docs
MLUtils.unsqueeze
MLUtils.flatten
MLUtils.stack
MLUtils.unstack
MLUtils.numobs
MLUtils.getobs
MLUtils.getobs!
MLUtils.chunk
MLUtils.group_counts
MLUtils.group_indices
MLUtils.batch
MLUtils.unbatch
MLUtils.batchseq
MLUtils.rpad(v::AbstractVector, n::Integer, p)
```
