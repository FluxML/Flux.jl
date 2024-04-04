# Training API Reference

The new version of Flux's training code was written as an independent package, [Optimisers.jl](https://github.com/FluxML/Optimisers.jl).
Only the function `train!` belongs to Flux itself.

The Optimisers package is designed to allow for immutable objects. But at present all Flux models contain parameter arrays (such as `Array`s and `CuArray`s) which can be updated in-place.
Because of this:

* The objects returned by `Optimisers.update!` can be ignored.
* Flux defines its own version of `setup` which checks this assumption.
  (Using instead `Optimisers.setup` will also work, they return the same thing.)

The available optimization rules are listed the [optimisation rules](@ref man-optimisers) page here. See the [Optimisers documentation](https://fluxml.ai/Optimisers.jl/dev/) for details on how the rules work.

```@docs
Flux.Train.setup
Flux.Train.train!(loss, model, data, state; cb)
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
