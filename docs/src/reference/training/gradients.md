```@meta
CollapsedDocStrings = true
```

# Automatic Differentiation in Flux

Flux's `gradient` function uses [Zygote](https://github.com/FluxML/Zygote.jl) by default, and also uses this function within [`train!`](@ref Flux.train!) to differentiate the model.
Zygote has its own [documentation](https://fluxml.ai/Zygote.jl/dev/), in particular listing some [important limitations](https://fluxml.ai/Zygote.jl/dev/limitations/).

Flux also has support for Enzyme.jl, documented [below](@ref autodiff-enzyme) and for Mooncake.jl.


```@docs
Flux.gradient(f, adtype::AbstractADType, args::Any...)
Flux.withgradient(f, adtype::AbstractADType, args::Any...)
```

## [Automatic Differentiation using Zygote.jl](@id autodiff-zygote)

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

## ChainRules for Zygote

Zygote uses [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) to define how to differentiate functions.

Sometimes it is necessary to exclude some code, or a whole function, from automatic differentiation. 
This can be done using the following methods:

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

Gradient customization for other AD packages such as Enzyme and Mooncake has to be done according to their own documentation.

## [Automatic Differentiation using Enzyme.jl](@id autodiff-enzyme)

[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is a new package for automatic differentiation.
Like Zygote.jl, calling `gradient(f, x)` causes it to hooks into the compiler and transform code that is executed while calculating `f(x)`, in order to produce code for `∂f/∂x`.
But it does so much later in the optimisation process (on LLVM instead of Julia's untyped IR) which you can [read about here](https://proceedings.nips.cc/paper/2020/file/9332c513ef44b682e9347822c2e457ac-Paper.pdf)].
It needs far fewer custom rules than Zygote/ChainRules, and in particular is able to support mutation of arrays.

Flux now builds in support for this, using Enzyme's own `Duplicated` type.
Calling `Duplicated` on any Flux model which was defined using `@layer` will allocate space for the gradient,
and passing that to `gradient` (or `withgradient`, or `train!`) will then use Enzyme instead of Zygote.
The gradient functions still return the gradient as usual, which can then be passed to `update!`:

```julia-repl
julia> using Flux, Enzyme

julia> model = Chain(Dense(28^2 => 32, sigmoid), Dense(32 => 10), softmax);  # from model zoo

julia> dup_model = Enzyme.Duplicated(model)  # this allocates space for the gradient
Duplicated(
  Chain(
    Dense(784 => 32, σ),                # 25_120 parameters
    Dense(32 => 10),                    # 330 parameters
    NNlib.softmax,
  ),
  # norm(∇) ≈ 0.0f0
)                   # Total: 4 arrays, 25_450 parameters, 199.391 KiB.

julia> x1 = randn32(28*28, 1);  # fake image

julia> y1 = [i==3 for i in 0:9];  # fake label

julia> grads_f = Flux.gradient((m,x,y) -> sum(abs2, m(x) .- y), dup_model, Const(x1), Const(y1))  # uses Enzyme
((layers = ((weight = Float32[-0.010354728 0.032972857 …
    -0.0014538406], σ = nothing), nothing),), nothing, nothing)
```

The gradient returned here is also stored within `dup_model`.
Both share the same arrays -- what is returned is not a copy, just a view of the same memory (wrapped in `NamedTuple`s instead of `struct`s).
They will all be set to zero when you call `gradient` again, then replaced with the new values.
Alternatively, `gradient(f, args...; zero=false)` will add the new gradient to what's already stored.

Writing `Const(x1)` is optional, just plain `x1` is implicitly constant.
Any set of `Duplicated` and `Const` arguments may appear in any order, so long as there is at least one `Duplicated`.

The gradient `grads_f[1]` can be passed to `update!` as usual.
But for convenience, you may also use what is stored within `Duplicated`.
These are equivalent ways to perform an update step:

```julia-repl
julia> opt_state = Flux.setup(Adam(), model)

julia> ans == Flux.setup(Adam(), dup_model)

julia> Flux.update!(opt_state, model, grads_f[1])  # exactly as for Zygote gradients

julia> Flux.update!(opt_state, dup_model)  # equivlent new path, Enzyme only
```

Instead of using these FLux functions, you can also use Enzyme's own functions directly.
`Enzyme.gradient` works like this:

```julia-repl
julia> grads_e = Enzyme.gradient(Reverse, (m,x,y) -> sum(abs2, m(x) .- y), model, Const(x1), Const(y1))
(Chain(Dense(784 => 32, σ), Dense(32 => 10), softmax), nothing, nothing)

julia> grads_f[1].layers[2].bias ≈ grads_e[1].layers[2].bias
true
```

Note that what `Enzyme.gradient` returns is an object like `deepcopy(model)` of the same type, `grads_e[1] isa Chain`.
But its fields contain the same gradient.


```@docs
Flux.gradient(f, args::Union{Flux.EnzymeCore.Const, Flux.EnzymeCore.Duplicated}...)
Flux.withgradient(f, args::Union{Flux.EnzymeCore.Const, Flux.EnzymeCore.Duplicated}...)
```

Enzyme.jl has [its own extensive documentation](https://enzymead.github.io/Enzyme.jl/stable/).


## Second-order AD

If you calculate a gradient within the loss function, then training will involve 2nd derivatives.
While this is in principle supported by Zygote.jl, there are many bugs, and Enzyme.jl is probably a better choice.
