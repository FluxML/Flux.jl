```@meta
CollapsedDocStrings = true
```

# [Loss Functions](@id man-losses)

Flux provides a large number of common loss functions used for training machine learning models.
They are grouped together in the `Flux.Losses` module.

Loss functions for supervised learning typically expect as inputs a target `y`, and a prediction `ŷ` from your model.
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

Losses compute in `Float32` when `ŷ` or `y` hold `Float16` or `BFloat16` numbers, such as the output
of a model converted with [`f16`](@ref Flux.f16) or [`bf16`](@ref Flux.bf16), and return a `Float32`.
Sums in low precision are inaccurate: a `BFloat16` sum stops growing once it is a few hundred times
larger than each term, and a `Float16` one overflows past 65504.
The gradients with respect to `ŷ` and `y` keep their element type, so that the rest of the backward
pass stays in low precision. The default `eps = eps(eltype(ŷ))` of some losses is then `eps(Float32)`.

## Function listing

```@docs
Flux.Losses.mae
Flux.Losses.mse
Flux.Losses.msle
Flux.Losses.huber_loss
Flux.Losses.label_smoothing
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
Flux.Losses.binary_focal_loss
Flux.Losses.focal_loss
Flux.Losses.siamese_contrastive_loss
ctc_loss
```
