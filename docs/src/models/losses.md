# Loss Functions

Flux provides a large number of common loss functions used for training machine learning models.
They are grouped together in the `Flux.Losses` module.

As an example, the crossentropy function for multi-class classification that takes logit predictions (i.e. not [`softmax`](@ref)ed)
can be imported with

```julia
using Flux.Losses: logitcrossentropy
```

Loss functions for supervised learning typically expect as inputs a true target `y` and a prediction `ŷ`.
In Flux's convention, the order of the arguments is the following:

```julia
loss(ŷ, y)
```

They are commonly passed as arrays of size `num_target_features x num_examples_in_batch`. 

Most loss functions in Flux have an optional argument `agg`, denoting the type of aggregation performed over the
batch:

```julia
loss(ŷ, y)                             # defaults to `mean`
loss(ŷ, y, agg = sum)                  # use `sum` for reduction
loss(ŷ, y, agg = x -> sum(x, dims=2))  # partial reduction
loss(ŷ, y, agg = x -> mean(w .* x))    # weighted mean
loss(ŷ, y, agg = identity)             # no aggregation.
```

## Losses Reference

```@autodocs
Modules = [Flux.Losses]
Pages   = ["functions.jl"]
```
