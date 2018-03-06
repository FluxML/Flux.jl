# Saving and Loading Models

You may wish to save models so that they can be loaded and run in a later
session. The easiest way to do this is via
[BSON.jl](https://github.com/MikeInnes/BSON.jl).

Save a model:

```julia
julia> using Flux

julia> model = Chain(Dense(10,5,relu),Dense(5,2),softmax)
Chain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)

julia> using BSON: @save

julia> @save "mymodel.bson" model
```

Load it again:

```julia
julia> using Flux

julia> using BSON: @load

julia> @load "mymodel.bson" model

julia> model
Chain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)
```

Models are just normal Julia structs, so it's fine to use any Julia storage
format for this purpose. BSON.jl is particularly well supported and most likely
to be forwards compatible (that is, models saved now will load in future
versions of Flux).

!!! note

    If a saved model's weights are stored on the GPU, the model will not load
    later on if there is no GPU support available. It's best to [move your model
    to the CPU](gpu.md) with `cpu(model)` before saving it.

## Saving Model Weights

In some cases it may be useful to save only the model parameters themselves, and
rebuild the model architecture in your code. You can use `params(model)` to get
model parameters. You can also use `data.(params)` to remove tracking.

```Julia
julia> using Flux

julia> model = Chain(Dense(10,5,relu),Dense(5,2),softmax)
Chain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)

julia> weights = Tracker.data.(params(model));

julia> using BSON: @save

julia> @save "mymodel.bson" weights
```

You can easily load parameters back into a model with `Flux.loadparams!`.

```julia
julia> using Flux

julia> model = Chain(Dense(10,5,relu),Dense(5,2),softmax)
Chain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)

julia> using BSON: @load

julia> @load "mymodel.bson" weights

julia> Flux.loadparams!(model, weights)
```

The new `model` we created will now be identical to the one we saved parameters for.

## Checkpointing

In longer training runs it's a good idea to periodically save your model, so that you can resume if training is interrupted (for example, if there's a power cut). You can do this by saving the model in the [callback provided to `train!`](training/training.md).

```julia
using Flux: throttle
using BSON: @save

m = Chain(Dense(10,5,relu),Dense(5,2),softmax)

evalcb = throttle(30) do
  # Show loss
  @save "model-checkpoint.bson" model
end
```

This will update the `"model-checkpoint.bson"` file every thirty seconds.

You can get more advanced by saving a series of models throughout training, for example

```julia
@save "model-$(now()).bson" model
```

will produce a series of models like `"model-2018-03-06T02:57:10.41.bson"`. You
could also store the current test set loss, so that it's easy to (for example)
revert to an older copy of the model if it starts to overfit.

```julia
bson("model-$(now()).bson", model = model, loss = testloss())
```
