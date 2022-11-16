# [Loss Functions](@id man-losses)

Flux provides a large number of common loss functions used for training machine learning models.
They are grouped together in the `Flux.Losses` module.

Loss functions for supervised learning typically expect as inputs a target `y`, and a prediction `ŷ` from your model.
In Flux's convention, the target is the last argumemt, so a new loss function could be defined:

```julia
newloss(ŷ, y) = sum(abs2, ŷ .- y)  # total squared error
```

All loss functions in Flux have a method which takes the model as the first argument, and calculates the prediction `ŷ = model(x)`.
This is convenient for [`train!`](@ref Flux.train)`(loss, model, [(x,y), (x2,y2), ...], opt)`.
For our example it could be defined:

```julia
newloss(model, x, y) = newloss(model(x), y)
```

Most loss functions in Flux have an optional keyword argument `agg`, which is the aggregation function used over the batch.
Thus you may call, for example:

```julia
crossentropy(ŷ, y)                           # defaults to `Statistics.mean`
crossentropy(ŷ, y; agg = sum)                # use `sum` instead
crossentropy(ŷ, y; agg = x->mean(w .* x))    # weighted mean
crossentropy(ŷ, y; agg = x->sum(x, dims=2))  # partial reduction, returns an array
```

### Function listing

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
```
