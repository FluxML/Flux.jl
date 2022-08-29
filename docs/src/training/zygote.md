# Automatic Differentiation using Zygote.jl

Flux re-exports the `gradient` from [Zygote](https://github.com/FluxML/Zygote.jl), and uses this function within [`train!`](@ref) to differentiate the model. Zygote has its own [documentation](https://fluxml.ai/Zygote.jl/dev/), in particulat listing some [limitations](https://fluxml.ai/Zygote.jl/dev/limitations/).

```@docs
Zygote.gradient
Zygote.jacobian
Zygote.withgradient
```

Sometimes it is necessary to exclude some code, or a whole function, from automatic differentiation. This can be done using [ChainRules](https://github.com/JuliaDiff/ChainRules.jl):

```@docs
ChainRulesCore.ignore_derivatives
ChainRulesCore.@non_differentiable
```

To manually supply the gradient for one function, you should define a method of `rrule`. ChainRules has [detailed documentation](https://juliadiff.org/ChainRulesCore.jl/stable/) on how this works.

```@docs
ChainRulesCore.rrule
```