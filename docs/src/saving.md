# Saving and Loading Models

You may wish to save models so that they can be loaded and run in a later
session. The easiest way to do this is via
[BSON.jl](https://github.com/MikeInnes/BSON.jl).

Save a model:

```jldoctest saving
julia> using Flux

julia> model = Chain(Dense(10, 5, NNlib.relu), Dense(5, 2), NNlib.softmax)
Chain(
  Dense(10 => 5, relu),                 # 55 parameters
  Dense(5 => 2),                        # 12 parameters
  NNlib.softmax,
)                   # Total: 4 arrays, 67 parameters, 524 bytes.

julia> using BSON: @save

julia> @save "mymodel.bson" model
```

Load it again:

```jldoctest saving
julia> using BSON: @load

julia> @load "mymodel.bson" model

julia> model
Chain(
  Dense(10 => 5, relu),                 # 55 parameters
  Dense(5 => 2),                        # 12 parameters
  NNlib.softmax,
)                   # Total: 4 arrays, 67 parameters, 524 bytes.

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
model parameters.

```jldoctest saving
julia> model = Chain(Dense(10 => 5,relu),Dense(5 => 2),softmax)
Chain(
  Dense(10 => 5, relu),                 # 55 parameters
  Dense(5 => 2),                        # 12 parameters
  NNlib.softmax,
)                   # Total: 4 arrays, 67 parameters, 524 bytes.

julia> weights = Flux.params(model);

julia> using BSON: @save

julia> @save "mymodel.bson" weights
```

You can easily load parameters back into a model with `Flux.loadparams!`.

```julia
julia> model = Chain(Dense(10 => 5,relu),Dense(5 => 2),softmax)
Chain(
  Dense(10 => 5, relu),                 # 55 parameters
  Dense(5 => 2),                        # 12 parameters
  NNlib.softmax,
)                   # Total: 4 arrays, 67 parameters, 524 bytes.

julia> using BSON: @load

julia> @load "mymodel.bson" weights

julia> Flux.loadparams!(model, weights)
```

The new `model` we created will now be identical to the one we saved parameters for.

## Checkpointing

In longer training runs it's a good idea to periodically save your model, so that you can resume if training is interrupted (for example, if there's a power cut). You can do this by saving the model in the [callback provided to `train!`](training/training.md).

```jldoctest saving
julia> using Flux: throttle

julia> using BSON: @save

julia> m = Chain(Dense(10 => 5, relu), Dense(5 => 2), softmax)
Chain(
  Dense(10 => 5, relu),                 # 55 parameters
  Dense(5 => 2),                        # 12 parameters
  NNlib.softmax,
)                   # Total: 4 arrays, 67 parameters, 524 bytes.

julia> evalcb = throttle(30) do
         # Show loss
         @save "model-checkpoint.bson" model
       end;
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
@save "model-$(now()).bson" model loss = testloss()
```

Note that to resume a model's training, you might need to restore other stateful parts of your training loop. Possible examples are stateful optimizers (which usually utilize an `IdDict` to store their state), and the randomness used to partition the original data into the training and validation sets.

You can store the optimiser state alongside the model, to resume training
exactly where you left off. BSON is smart enough to [cache values](https://github.com/JuliaIO/BSON.jl/blob/v0.3.4/src/write.jl#L71) and insert links when saving, but only if it knows everything to be saved up front. Thus models and optimizers must be saved together to have the latter work after restoring.

```julia
opt = ADAM()
@save "model-$(now()).bson" model opt
```

