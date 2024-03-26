# Saving and Loading Models

You may wish to save models so that they can be loaded and run in a later
session. Flux provides a number of ways to do this. 
The recommended way, which is the most robust one for long term storage, 
is to use [`Flux.state`](@ref) in combination with a serialization format like
[JLD2.jl](https://juliaio.github.io/JLD2.jl/dev/) or
[BSON.jl](https://github.com/JuliaIO/BSON.jl).

Save a model:

```jldoctest saving
julia> using Flux

julia> struct MyModel
           net
       end

julia> Flux.@layer MyModel

julia> MyModel() = MyModel(Chain(Dense(10 => 5, relu), Dense(5 => 2)));

julia> model = MyModel()
MyModel(Chain(Dense(10 => 5, relu), Dense(5 => 2)))  # 67 parameters

julia> model_state = Flux.state(model);

julia> using JLD2

julia> jldsave("mymodel.jld2"; model_state)
```

Load it again in a new session using [`Flux.loadmodel!`](@ref):

```jldoctest saving
julia> using Flux, JLD2

julia> model_state = JLD2.load("mymodel.jld2", "model_state");

julia> model = MyModel(); # MyModel definition must be available

julia> Flux.loadmodel!(model, model_state);
```

!!! note

    If a saved model's parameters are stored on the GPU, the model will not load
    later on if there is no GPU support available. It's best to [move your model
    to the CPU](gpu.md) with `cpu(model)` before saving it.


## Checkpointing

In longer training runs it's a good idea to periodically save your model, so that you can resume if training is interrupted (for example, if there's a power cut). 

```jldoctest saving
julia> using Flux: throttle

julia> using JLD2

julia> m = Chain(Dense(10 => 5, relu), Dense(5 => 2))
Chain(
  Dense(10 => 5, relu),                 # 55 parameters
  Dense(5 => 2),                        # 12 parameters
)                   # Total: 4 arrays, 67 parameters, 524 bytes.

julia> for epoch in 1:10
          # ... train model ...
          jldsave("model-checkpoint.jld2", model_state = Flux.state(m))
       end;
```

This will update the `"model-checkpoint.jld2"` every epoch.

You can get more advanced by saving a series of models throughout training, for example

```julia
jldsave("model-$(now()).jld2", model_state = Flux.state(m))
```

will produce a series of models like `"model-2018-03-06T02:57:10.41.jld2"`. You
could also store the current test set loss, so that it's easy to (for example)
revert to an older copy of the model if it starts to overfit.

```julia
jldsave("model-$(now()).jld2", model_state = Flux.state(m), loss = testloss())
```

Note that to resume a model's training, you might need to restore other stateful parts of your training loop. Possible examples are the optimiser state and the randomness used to partition the original data into the training and validation sets.

You can store the optimiser state alongside the model, to resume training
exactly where you left off: 

```julia
model = MyModel()
opt_state = Flux.setup(AdamW(), model)

# ... train model ...

model_state = Flux.state(model)
jldsave("checkpoint_epoch=42.jld2"; model_state, opt_state)
```

# Saving Models as Julia Structs

Models are just normal Julia structs, so it's fine to use any Julia storage
format to save the struct as it is instead of saving the state returned by [`Flux.state`](@ref). 
[BSON.jl](https://github.com/JuliaIO/BSON.jl) is particularly convenient for this,
since it can also save anonymous functions, which are sometimes part of a model definition.

Save a model:

```jldoctest saving
julia> using Flux

julia> model = Chain(Dense(10, 5, NNlib.relu), Dense(5, 2));

julia> using BSON: @save

julia> @save "mymodel.bson" model
```

Load it again in a new session:

```jldoctest saving
julia> using Flux, BSON

julia> BSON.@load "mymodel.bson" model

julia> model
Chain(
  Dense(10 => 5, relu),                 # 55 parameters
  Dense(5 => 2),                        # 12 parameters
)                   # Total: 4 arrays, 67 parameters, 524 bytes.
```
!!! warning
    Saving models this way could lead to compatibility issues across julia versions
    and across Flux versions if some of the Flux layers' internals are changed.
    It is therefore not recommended for long term storage, use [`Flux.state`](@ref) instead.

!!! warning

    Previous versions of Flux suggested saving only the model weights using
    `@save "mymodel.bson" params(model)`.
    This is no longer recommended and even strongly discouraged.
    Saving models this way will only store the trainable parameters which
    will result in incorrect behavior for layers like `BatchNorm`.
