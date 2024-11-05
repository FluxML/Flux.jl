
# [Automatic Differentiation using Enzyme.jl](@id autodiff-enzyme)

[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is a new package for automatic differentiation.
Like Zygote.jl, calling `gradient(f, x)` causes it to hooks into the compiler and transform code that is executed while calculating `f(x)`, in order to produce code for `∂f/∂x`.
But it does so much later in the optimisation process (on LLVM instead of Julia's untyped IR).
It needs far fewer custom rules than Zygote/ChainRules, and in particular is able to support mutation of arrays.

Flux now builds in support for this, using Enzyme's own `Duplicated` type.
Calling `Duplicated` on any Flux model which was defined using `@layer` will allocate space for the gradient,
and passing that to `gradient` (or `withgradient`, or `train!`) will then use Enzyme instead of Zygote.
The gradient functions still return the gradient as usual, which can then be passed to `update!`:

```julia
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

julia> grads_f = Flux.gradient((m,x,y) -> sum(abs2, m(x) .- y), dup_model, Const(x1), Const(y1))
((layers = ((weight = Float32[-0.010354728 0.032972857 …
    -0.0014538406], σ = nothing), nothing),), nothing, nothing)
```

The gradient returned here is also stored within `dup_model`, it shares the same arrays. It will be set to zero when you call `gradient` again.

Writing `Const(x1)` is optional, just plain `x1` is implicitly constant.
Any set of `Duplicated` and `Const` arguments may appear in any order, so long as there is at least one `Duplicated`.

Instead of using these FLux functions, you can also use Enzyme's own functions directly.
`Enzyme.gradient` works like this:

```julia
julia> grads_e = Enzyme.gradient(Reverse, (m,x,y) -> sum(abs2, m(x) .- y), model, Const(x1), Const(y1))
(Chain(Dense(784 => 32, σ), Dense(32 => 10), softmax), nothing, nothing)

julia> grads_f[1].layers[2].bias ≈ grads_e[1].layers[2].bias
true
```

Note that what `Enzyme.gradient` returns is an object like `deepcopy(model)` of the same type, `grads_e[1] isa Chain`.
But its fields contain the same gradient.

There is also a method of `train!` which similarly takes `Duplicated(model)`:

```julia
julia> opt_state = Flux.setup(Adam(0), model);

julia> Flux.train!((m,x,y) -> sum(abs2, m(x) .- y), dup_model, [(x1, y1)], opt_state)
```



## Listing

```@docs
Flux.gradient(f, args::Union{EnzymeCore.Const, EnzymeCore.Duplicated}...)
Flux.withgradient(f, args::Union{EnzymeCore.Const, EnzymeCore.Duplicated}...)
Flux.train!(loss, model::EnzymeCore.Duplicated, data, opt)
```
