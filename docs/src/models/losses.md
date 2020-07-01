## Loss Functions

Flux provides a large number of common loss functions used for training machine learning models.

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

### Losses Reference

```@docs
Flux.mae
Flux.mse
Flux.msle
Flux.huber_loss
Flux.crossentropy
Flux.logitcrossentropy
Flux.bce_loss
Flux.logitbce_loss
Flux.kldivergence
Flux.poisson_loss
Flux.hinge_loss
Flux.squared_hinge_loss
Flux.dice_coeff_loss
Flux.tversky_loss
```
