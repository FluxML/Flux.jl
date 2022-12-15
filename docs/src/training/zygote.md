# Automatic Differentiation using Zygote.jl

Flux re-exports the `gradient` from [Zygote](https://github.com/FluxML/Zygote.jl), and uses this function within [`train!`](@ref Flux.train!) to differentiate the model. Zygote has its own [documentation](https://fluxml.ai/Zygote.jl/dev/), in particular listing some [important limitations](https://fluxml.ai/Zygote.jl/dev/limitations/).


## Explicit style

The preferred way of using Zygote, and the only way of using most other AD packages,
is to explicitly provide a function and its arguments.

```@docs
Zygote.gradient(f, args...)
Zygote.withgradient(f, args...)
Zygote.jacobian(f, args...)
Zygote.withgradient
```

## Implicit style (Flux ≤ 0.13)

Flux used to use what Zygote calls "implicit" gradients, [described here](https://fluxml.ai/Zygote.jl/dev/#Explicit-and-Implicit-Parameters-1) in its documentation.
However, support for this will be removed from Flux 0.14.

!!! compat "Training"
    The blue-green boxes in the [training section](@ref man-training) describe
    the changes needed to upgrade old code from implicit to explicit style.

```@docs
Zygote.gradient
Zygote.Params
Zygote.Grads
Zygote.jacobian(loss, ::Params)
```

## ChainRules

Sometimes it is necessary to exclude some code, or a whole function, from automatic differentiation. This can be done using [ChainRules](https://github.com/JuliaDiff/ChainRules.jl):

```@docs
ChainRulesCore.ignore_derivatives
ChainRulesCore.@non_differentiable
```

To manually supply the gradient for one function, you should define a method of `rrule`. ChainRules has [detailed documentation](https://juliadiff.org/ChainRulesCore.jl/stable/) on how this works.

```@docs
ChainRulesCore.rrule
```
