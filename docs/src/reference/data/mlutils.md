```@meta
CurrentModule = Flux
CollapsedDocStrings = true
```

# Working with Data, using MLUtils.jl

Flux re-exports the `DataLoader` type and utility functions for working with
data from [MLUtils](https://github.com/JuliaML/MLUtils.jl).

## `DataLoader`

The `DataLoader` can be used to create mini-batches of data, in the format [`train!`](@ref Flux.train!) expects.

```@docs
MLUtils.DataLoader
```

## Utility Functions

The utility functions are meant to be used while working with data;
these functions help create inputs for your models or batch your dataset.

```@docs
MLUtils.batch
MLUtils.batchsize
MLUtils.batchseq
MLUtils.batch_sequence
MLUtils.BatchView
MLUtils.chunk
MLUtils.eachobs
MLUtils.fill_like
MLUtils.filterobs
Flux.flatten
MLUtils.flatten
MLCore.getobs
MLCore.getobs!
MLUtils.joinobs
MLUtils.group_counts
MLUtils.group_indices
MLUtils.groupobs
MLUtils.kfolds
MLUtils.leavepout
MLUtils.mapobs
MLCore.numobs
MLUtils.normalise
MLUtils.obsview
MLUtils.ObsView
MLUtils.ones_like
MLUtils.oversample
MLUtils.randobs
MLUtils.rand_like
MLUtils.randn_like
MLUtils.rpad_constant
MLUtils.shuffleobs
MLUtils.splitobs
MLUtils.unbatch
MLUtils.undersample
MLUtils.unsqueeze
MLUtils.unstack
MLUtils.zeros_like
```
