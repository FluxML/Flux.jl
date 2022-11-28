# Training API Reference

The new version of Flux's training code was written as an independent package, [Optimisers.jl](https://github.com/FluxML/Optimisers.jl).
Only the function `train!` belongs to Flux itself.

The Optimisers package is designed to allow for immutable objects. But at present all Flux models contain parameter arrays (such as `Array`s and `CuArray`s) which can be updated in-place.
Because of this:

* The objects returned by `Optimisers.update!` can be ignored.
* Flux defines its own version of `setup` which checks this assumption.
  (Using instead `Optimisers.setup` will also work, they return the same thing.)

The new implementation of rules such as Adam in the Optimisers is quite different from the old one in `Flux.Optimise`. In Flux 0.13, `Flux.Adam()` returns the old one, with supertype `Flux.Optimise.AbstractOptimiser`, but `setup` will silently translate it to its new counterpart.
The available rules are listed the [optimisation rules](@ref man-optimisers) page here;
see the [Optimisers documentation](https://fluxml.ai/Optimisers.jl/dev/) for details on how the new rules work.

```@docs
Flux.Train.setup
Flux.Train.train!(loss, model, data, opt; cb)
Optimisers.update!
```

`train!` uses [`@progress`](https://github.com/JuliaLogging/ProgressLogging.jl) which should show a progress bar in VSCode automatically.
To see one in a terminal, you will need to install [TerminalLoggers.jl](https://github.com/JuliaLogging/TerminalLoggers.jl)
and follow its setup instructions.

## Optimisation Modifiers

The state returned by `setup` can be modified to temporarily prevent training of
some parts of the model, or to change the learning rate or other hyperparameter.
The functions for doing so may be accessed as `Flux.freeze!`, `Flux.thaw!`, and `Flux.adjust!`.
All mutate the state (or part of it) and return `nothing`.

```@docs
Optimisers.adjust!
Optimisers.freeze!
Optimisers.thaw!
```

## Implicit style (Flux ≤ 0.13)

Flux used to handle gradients, training, and optimisation rules quite differently.
The new style described above is called "explicit" by Zygote, and the old style "implicit".
Flux 0.13 is the transitional version which supports both; Flux 0.14 will remove the old.

!!! compat "How to upgrade"
    The blue-green boxes in the [training section](@ref man-training) describe
    the changes needed to upgrade old code.

For full details on the interface for implicit-style optimisers, see the [Flux 0.13.6 manual](https://fluxml.ai/Flux.jl/v0.13.6/training/training/).

```@docs
Flux.params
Flux.Optimise.update!(opt::Flux.Optimise.AbstractOptimiser, xs::Flux.Params, gs)
Flux.Optimise.train!(loss, ps::Flux.Params, data, opt::Flux.Optimise.AbstractOptimiser; cb)
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

Implicit `train!` takes an additional argument, `cb`, that's used for callbacks so that you can observe the training process. For example:

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

See the page about [callback helpers](@ref man-callback-helpers) for more.

