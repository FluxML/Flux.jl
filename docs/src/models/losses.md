# Loss Functions

Flux provides a large number of common loss functions used for training machine learning models.
They are grouped together in the `Flux.Losses` module.

Loss functions for supervised learning typically expect as inputs a target `y`, and a prediction `ŷ`.
In Flux's convention, the order of the arguments is the following

```julia
loss(ŷ, y)
```

Most loss functions in Flux have an optional argument `agg`, denoting the type of aggregation performed over the
batch:

```julia
loss(ŷ, y)                         # defaults to `mean`
loss(ŷ, y, agg=sum)                # use `sum` for reduction
loss(ŷ, y, agg=x->sum(x, dims=2))  # partial reduction
loss(ŷ, y, agg=x->mean(w .* x))    # weighted mean
loss(ŷ, y, agg=identity)           # no aggregation.
```

## Losses Reference

```@docs
Flux.Losses.mae
Flux.Losses.mse
Flux.Losses.msle
Flux.Losses.huber_loss
Flux.Losses.crossentropy
Flux.Losses.logitcrossentropy
Flux.Losses.binarycrossentropy
Flux.Losses.logitbinarycrossentropy
Flux.Losses.kldivergence
Flux.Losses.poisson_loss
Flux.Losses.hinge_loss
Flux.Losses.squared_hinge_loss
Flux.Losses.dice_coeff_loss
Flux.Losses.tversky_loss
```
