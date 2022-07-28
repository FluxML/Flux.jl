# Utility Functions

Flux provides utility functions which can be used to initialize your layers
or to regularly execute callback functions.

## Layer Initialisation

Flux initialises convolutional layers and recurrent cells with `glorot_uniform` by default.
Most layers accept a function as an `init` keyword, which replaces this default. For example:

```jldoctest; setup = :(using Flux)
julia> conv = Conv((3, 3), 3 => 2, relu; init=Flux.glorot_normal)
Conv((3, 3), 3 => 2, relu)  # 56 parameters

julia> conv.bias
2-element Vector{Float32}:
 0.0
 0.0
```

Note that `init` creates the weight array, but not the bias vector.

Many of the initialisation functions accept keywords such as `gain`, 
and a random number generator. To make it easy to pass these to layers,
there are methods which return a function:

```jldoctest; setup = :(using Flux, Random)
julia> Dense(4 => 5, tanh; init=Flux.glorot_uniform(gain=2))
Dense(4 => 5, tanh)  # 25 parameters

julia> Dense(4 => 5, tanh; init=Flux.randn32(MersenneTwister(1)))
Dense(4 => 5, tanh)  # 25 parameters
```

```@docs
Flux.glorot_uniform
Flux.glorot_normal
Flux.kaiming_uniform
Flux.kaiming_normal
Flux.truncated_normal
Flux.orthogonal
Flux.sparse_init
Flux.identity_init
Flux.ones32
Flux.rand32
```

## Changing the type of model parameters

The default `eltype` for models is `Float32` since models are often trained/run on GPUs.
The `eltype` of model `m` can be changed to `Float64` by `f64(m)`:

```@docs
Flux.f64
Flux.f32
```

## Model Building

Flux provides some utility functions to help you generate models in an automated fashion.

[`Flux.outputsize`](@ref) enables you to calculate the output sizes of layers like [`Conv`](@ref)
when applied to input samples of a given size. This is achieved by passing a "dummy" array into
the model that preserves size information without running any computation.
`outputsize(f, inputsize)` works for all layers (including custom layers) out of the box.
By default, `inputsize` expects the batch dimension,
but you can exclude the batch size with `outputsize(f, inputsize; padbatch=true)` (assuming it to be one).

Using this utility function lets you automate model building for various inputs like so:
```julia
"""
    make_model(width, height, inchannels, nclasses;
               layer_config = [16, 16, 32, 32, 64, 64])

Create a CNN for a given set of configuration parameters.

# Arguments
- `width`: the input image width
- `height`: the input image height
- `inchannels`: the number of channels in the input image
- `nclasses`: the number of output classes
- `layer_config`: a vector of the number of filters per each conv layer
"""
function make_model(width, height, inchannels, nclasses;
                    layer_config = [16, 16, 32, 32, 64, 64])
  # construct a vector of conv layers programmatically
  conv_layers = [Conv((3, 3), inchannels => layer_config[1])]
  for (infilters, outfilters) in zip(layer_config, layer_config[2:end])
    push!(conv_layers, Conv((3, 3), infilters => outfilters))
  end

  # compute the output dimensions for the conv layers
  # use padbatch=true to set the batch dimension to 1
  conv_outsize = Flux.outputsize(conv_layers, (width, height, nchannels); padbatch=true)

  # the input dimension to Dense is programatically calculated from
  #  width, height, and nchannels
  return Chain(conv_layers..., Dense(prod(conv_outsize) => nclasses))
end
```

```@docs
Flux.outputsize
```

## Model Abstraction

```@docs
Flux.modules
Flux.nfan
```

## Callback Helpers

```@docs
Flux.throttle
Flux.stop
Flux.skip
```

## Patience Helpers

Flux provides utilities for controlling your training procedure according to some monitored condition and a maximum `patience`. For example, you can use `early_stopping` to stop training when the model is converging or deteriorating, or you can use `plateau` to check if the model is stagnating.

For example, below we create a pseudo-loss function that decreases, bottoms out, then increases. The early stopping trigger will break the loop before the loss increases too much.
```julia
# create a pseudo-loss that decreases for 4 calls, then starts increasing
# we call this like loss()
loss = let t = 0
  () -> begin
    t += 1
    (t - 4) ^ 2
  end
end

# create an early stopping trigger
# returns true when the loss increases for two consecutive steps
es = early_stopping(loss, 2; init_score = 9)

# this will stop at the 6th (4 decreasing + 2 increasing calls) epoch
@epochs 10 begin
  es() && break
end
```

The keyword argument `distance` of `early_stopping` is a function of the form `distance(best_score, score)`. By default `distance` is `-`, which implies that the monitored metric `f` is expected to be decreasing and mimimized. If you use some increasing metric (e.g. accuracy), you can customize the `distance` function: `(best_score, score) -> score - best_score`.
```julia
# create a pseudo-accuracy that increases by 0.01 each time from 0 to 1
# we call this like acc()
acc = let v = 0
  () -> v = max(1, v + 0.01)
end

# create an early stopping trigger for accuracy
es = early_stopping(acc, 3; delta = (best_score, score) -> score - best_score)

# this will iterate until the 10th epoch
@epochs 10 begin
  es() && break
end
```

`early_stopping` and `plateau` are both built on top of `patience`. You can use `patience` to build your own triggers that use a patient counter. For example, if you want to trigger when the loss is below a threshold for several consecutive iterations:
```julia
threshold(f, thresh, delay) = patience(delay) do
  f() < thresh
end
```

Both `predicate` in `patience` and `f` in `early_stopping` / `plateau` can accept extra arguments. You can pass such extra arguments to `predicate` or `f` through the returned function:
```julia
trigger = patience((a; b) -> a > b, 3)

# this will iterate until the 10th epoch
@epochs 10 begin
  trigger(1; b = 2) && break
end

# this will stop at the 3rd epoch
@epochs 10 begin
  trigger(3; b = 2) && break
end
```

```@docs
Flux.patience
Flux.early_stopping
Flux.plateau
```
