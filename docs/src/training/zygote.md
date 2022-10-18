# Automatic Differentiation using Zygote.jl

Flux re-exports the `gradient` from [Zygote](https://github.com/FluxML/Zygote.jl), and uses this function within [`train!`](@ref Flux.train!) to differentiate the model. Zygote has its own [documentation](https://fluxml.ai/Zygote.jl/dev/), in particular listing some [important limitations](https://fluxml.ai/Zygote.jl/dev/limitations/).

## Implicit style

Flux uses primarily what Zygote calls "implicit" gradients, [described here](https://fluxml.ai/Zygote.jl/dev/#Explicit-and-Implicit-Parameters-1) in its documentation. 

```@docs
Zygote.gradient
Zygote.Params
Zygote.Grads
Zygote.jacobian(loss, ::Params)
```

## Explicit style

The other way of using Zygote, and using most other AD packages, is to explicitly provide a function and its arguments.

```@docs
Zygote.gradient(f, args...)
Zygote.withgradient(f, args...)
Zygote.jacobian(f, args...)
Zygote.withgradient
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
