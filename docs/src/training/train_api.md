# Training API


```@docs
Flux.Train.setup
Flux.Train.update!
Flux.Train.train!
```

## Implicit style

Flux used to handle gradients, training, and optimisation rules quite differently.
The new style described above is called "explicit" by Zygote, and the old style "implicit".
Flux 0.13 is the transitional version which supports both.

For full details on how to use the implicit style, see [Flux 0.13.6 manual](https://fluxml.ai/Flux.jl/v0.13.6/training/training/).


```@docs
Flux.params
Flux.Optimise.update!
Flux.Optimise.train!
```


Note that, by default, `train!` only loops over the data once (a single "epoch").
A convenient way to run multiple epochs from the REPL is provided by `@epochs`.

```julia
julia> using Flux: @epochs

julia> @epochs 2 println("hello")
[ Info: Epoch 1
hello
[ Info: Epoch 2
hello

julia> @epochs 2 Flux.train!(...)
# Train for two epochs
```

```@docs
Flux.@epochs
```

## Callbacks

`train!` takes an additional argument, `cb`, that's used for callbacks so that you can observe the training process. For example:

```julia
train!(objective, ps, data, opt, cb = () -> println("training"))
```

Callbacks are called for every batch of training data. You can slow this down using `Flux.throttle(f, timeout)` which prevents `f` from being called more than once every `timeout` seconds.

A more typical callback might look like this:

```julia
test_x, test_y = # ... create single batch of test data ...
evalcb() = @show(loss(test_x, test_y))
throttled_cb = throttle(evalcb, 5)
Flux.@epochs 20 Flux.train!(objective, ps, data, opt, cb = throttled_cb)
```

Calling `Flux.stop()` in a callback will exit the training loop early.

```julia
cb = function ()
  accuracy() > 0.9 && Flux.stop()
end
```
