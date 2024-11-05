```@meta
CollapsedDocStrings = true
```

# [Automatic Differentiation using Zygote.jl](@id autodiff-zygote)

Flux's `gradient` function uses [Zygote](https://github.com/FluxML/Zygote.jl) by default, and also uses this function within [`train!`](@ref Flux.train!) to differentiate the model.
Zygote has its own [documentation](https://fluxml.ai/Zygote.jl/dev/), in particular listing some [important limitations](https://fluxml.ai/Zygote.jl/dev/limitations/).

Flux also has support for Enzyme.jl, documented [on its own page](@ref autodiff-enzyme).

## Explicit style

The preferred way of using Zygote, and the only way of using most other AD packages,
is to explicitly provide a function and its arguments.

```@docs
Zygote.gradient(f, args...)
Zygote.withgradient(f, args...)
Zygote.jacobian(f, args...)
Zygote.withjacobian(f, args...)
Zygote.hessian
Zygote.hessian_reverse
Zygote.diaghessian
Zygote.pullback
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
ChainRulesCore.frule
ChainRulesCore.@scalar_rule
ChainRulesCore.NoTangent
ChainRulesCore.ZeroTangent
ChainRulesCore.RuleConfig
ChainRulesCore.Tangent
ChainRulesCore.canonicalize
```
