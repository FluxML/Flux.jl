# MLUtils.jl

Flux re-exports the `DataLoader` type and utility functions for working with
data from [MLUtils](https://github.com/JuliaML/MLUtils.jl).

## DataLoader

`DataLoader` can be used to handle iteration over mini-batches of data.

```@docs
MLUtils.DataLoader
```

## Utility functions for working with data

The utility functions are meant to be used while working with data;
these functions help create inputs for your models or batch your dataset.

Below is a non-exhaustive list of such utility functions.

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
