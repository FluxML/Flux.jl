# The Gradient Zoo

The heart of how deep learning works is backpropagation of error,
also known as reverse-mode automatic differentiation.
Given a model, some data, and a loss function, this answers the question
"what direction, in the space of the model's parameters, reduces the loss fastest?"

### `gradient(f, x)` interface

Julia's ecosystem has many versions of `gradient(f, x)`, which evaluates `y = f(x)` then retuns `∂y_∂x`. The details of how they do this vary, but the interfece is similar. An incomplete list is (alphabetically):

```julia
julia> Diffractor.gradient(x -> sum(sqrt, x), [1 4 16.])
([0.5 0.25 0.125],)

julia> Enzyme.gradient(Reverse, x -> sum(sqrt, x), [1 4 16.])
1×3 Matrix{Float64}:
 0.5  0.25  0.125

julia> ForwardDiff.gradient(x -> sum(sqrt, x), [1 4 16.])
1×3 Matrix{Float64}:
 0.5  0.25  0.125

julia> ReverseDiff.gradient(x -> sum(sqrt, x), [1 4 16.])
1×3 Matrix{Float64}:
 0.5  0.25  0.125

julia> DifferentiationInterface.gradient(x -> sum(sqrt, x), AutoTapir(), [1 4 16.])
1×3 Matrix{Float64}:
 0.5  0.25  0.125

julia> Tracker.gradient(x -> sum(sqrt, x), [1 4 16.])
([0.5 0.25 0.125] (tracked),)

julia> Yota.grad(x -> sum(sqrt, x), [1 4 16.])
(7.0, (ChainRulesCore.ZeroTangent(), [0.5 0.25 0.125]))

julia> Zygote.withgradient(x -> sum(sqrt, x), [1 4 16.])
(val = 7.0, grad = ([0.5 0.25 0.125],))
```

These all show the same `∂y_∂x` with respect to `x::Vector`. Sometimes, the result is within a tuple or a NamedTuple, containing `y` as well as the gradient.

Note that in all cases, only code executed within the call to `gradient` is differentiated. Calculating the objective function before calling `gradient` will not work, as all information about the steps from `x` to `y` has been lost. For example:

```julia
julia> y = sum(sqrt, x)  # calculate the forward pass alone
7.0

julia> y isa Float64  # has forgotten about sqrt and sum
true

julia> Zygote.gradient(x -> y, x)  # this cannot work, and gives zero
(nothing,)
```

### `gradient(f, model)` for Flux models

However, the parameters of a Flux model are encapsulated inside the various layers. The model is a set of nested structures, and the gradients `∂loss_∂model` which Flux uses are similarly nested objects.
For example, let's set up a simple model & loss:

```julia
julia> model = Chain(Embedding(reshape(1:6, 2,3) .+ 0.0), softmax)
Chain(
  Embedding(3 => 2),                    # 6 parameters
  NNlib.softmax,
) 

julia> model.layers[1].weight  # this is the wrapped parameter array
2×3 Matrix{Float64}:
 1.0  3.0  5.0
 2.0  4.0  6.0

julia> loss(m) = sum(abs2, m(1))
loss (generic function with 3 methods)

julia> loss(model)  # returns a number
0.6067761335170363
```

Then we can find the same gradient using several packages:

```julia
julia> val, grads_z = Zygote.withgradient(loss, model)
(val = 0.6067761335170363, grad = ((layers = ((weight = [-0.18171549534589682 0.0 0.0; 0.18171549534589682 0.0 0.0],), nothing),),))

julia> _, grads_t = Tracker.withgradient(loss, model)
(val = 0.6067761335170363, grad = ((layers = ((weight = [-0.18171549534589682 0.0 0.0; 0.18171549534589682 0.0 0.0],), nothing),),))

julia> grads_d = Diffractor.gradient(loss, model)
(Tangent{Chain{Tuple{Embedding{Matrix{Float64}}, typeof(softmax)}}}(layers = (Tangent{Embedding{Matrix{Float64}}}(weight = [-0.18171549534589682 0.0 0.0; 0.18171549534589682 0.0 0.0],), ChainRulesCore.NoTangent()),),)

julia> grad_e = Enzyme.gradient(Reverse, loss, model)
Chain(
  Embedding(3 => 2),                    # 6 parameters
  NNlib.softmax,
) 
```

While the type returned for `∂loss_∂model` varies, they all have the same nested structure, matching that of the model. This is all that Flux needs.

```julia
julia> grads_z[1].layers[1].weight  # get the weight matrix
2×3 Matrix{Float64}:
 -0.181715  0.0  0.0
  0.181715  0.0  0.0

julia> grad_e.layers[1].weight  # get the corresponding gradient matrix
2×3 Matrix{Float64}:
 -0.181715  0.0  0.0
  0.181715  0.0  0.0
```

Here's Flux updating the model using each gradient:

```julia
julia> opt = Flux.setup(Descent(1/3), model)
(layers = ((weight = Leaf(Descent(0.333333), nothing),), ()),)

julia> model_z = deepcopy(model);

julia> Flux.update!(opt, model_z, grads_z[1]);

julia> model_z.layers[1].weight  # updated weight matrix
2×3 Matrix{Float64}:
 1.06057  3.0  5.0
 1.93943  4.0  6.0

julia> model_e = deepcopy(model);

julia> Flux.update!(opt, model_e, grad_e)[2][1].weight  # same update
2×3 Matrix{Float64}:
 1.06057  3.0  5.0
 1.93943  4.0  6.0
```

In this case they are all identical, but there are some caveats, explored below.

<hr/>

## Automatic Differentiation Packages

Both Zygote and Tracker were written for Flux, and at present, Flux loads Zygote and exports `Zygote.gradient`, and calls this within `Flux.train!`. But apart from that, there is very little coupling between Flux and the automatic differentiation package.

This page has very brief notes on how all these packages compare, as a guide for anyone wanting to experiment with them. We stress "experiment" since Zygote is (at present) by far the best-tested.

### [Zygote.jl](https://github.com/FluxML/Zygote.jl/issues)

Source-to-source, within Julia. 

* By far the best-tested option for Flux models.

* Long compilation times, on the first call.

* Allows mutation of structs, but not of arrays. This leads to the most common error... sometimes this happens because you mutate an array, often because you call some function which, internally, creates the array it wants to return & then fills it in.

* Custom rules via `ZygoteRules.@adjpoint` or better, `ChainRulesCore.rrule`.

* Returns nested NamedTuples and Tuples, and uses `nothing` to mean zero.


!!! compat "Deprecated: Zygote's implicit mode"
    Flux's default used to be work like this, instead of using deeply nested trees for gradients as above:
    ```julia
    julia> ps = Flux.params(model)  # dictionary-like object, with global `objectid` refs
    Params([Float32[1.0 3.0 5.0; 2.0 4.0 6.0]])

    julia> val, grad = Zygote.withgradient(() -> loss(model), ps)
    (val = 0.6067761f0, grad = Grads(...))

    julia> grad[model.layers[1].weight]  # another dictionary, indexed by parameter arrays
    2×3 Matrix{Float32}:
     0.0  0.0  -0.181715
     0.0  0.0   0.181715
    ```
    The code inside Zygote is much the same -- do not expect large changes in speed, nor any changes in what works and what does not.

### [Tracker.jl](https://github.com/FluxML/Tracker.jl)

Uses a `TrackedArray` type to build a tape. The recommended interface `Tracker.withgradient` hides this, and works much like the Zygote one. Notice in particular that this cannot work:

```julia
julia> val = loss(model)  # computed outside gradient context
0.6067761f0

julia> Tracker.withgradient(_ -> val, model)  # this won't work!
(val = 0.6067761f0, grad = (nothing,))
```

Can be used in lower-level ways which directly expose the tracked types:

```julia
julia> model_tracked = Flux.fmap(x -> x isa Array ? Tracker.param(x) : x, model)
Chain(
  Embedding(3 => 2),                    # 6 parameters
  NNlib.softmax,
) 

julia> val_tracked = loss(model_tracked)
0.6067761f0 (tracked)

julia> Tracker.back!(val_tracked)

julia> model_tracked.layers[1].weight.grad
2×3 Matrix{Float32}:
 0.0  0.0  -0.181715
 0.0  0.0   0.181715
```

* Quick to run, on the first call.

* Generally slower than Zygote, allocates more, and supports fewer operations.

* Custom rules via its own `track` and `@grad`.


### [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)

New package which works on the LLVM code which Julia compiles down to.

* Allows mutation of arrays.

* Long compilation times, on the first call.

* Does not at present work on all Flux models, due to missing rules.

* Does not always handle type instability.

* Custom rules by its own rules... Generally fewer such rules than Zygote, and at a lower level -- applied to `BLAS.gemm!` not `*`.

* Returns another struct of the same type as the model, such as `Chain` above. Non-differentiable objects are left alone, not replaced by a zero.

### Tapir.jl

Another new AD to watch. Many similariries in its approach to Enzyme.jl, but operates all in Julia.


### [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl)

To first approximation, Diffractor may be thought of as a re-write of Zygote, aiming to reduce compilation times, and to handle higher-order derivatives much better.

At present, development is focused on the forward-mode part. Reverse-mode `gradient` exists,
but fails on many Flux models.

* Custom rules via `ChainRulesCore.rrule`.

* Returns nested `Tangent` types, from ChainRulesCore, with zeros indicated by `NoTangent()`.


### [Yota.jl](https://github.com/dfdx/Yota.jl)

Another Julia source-to-source reverse-mode AD.

* Does not work on Julia 1.10

* Does not handle branches based on runtime values, due to how its tape works.

* Custom rules via `ChainRulesCore.rrule`.

* Returns nested `Tangent` types, from ChainRulesCore


### [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)

Forward mode is a different algorithm... 

* Needs a flat vector

* Forward mode is generally not what you want!

* `gradient(f, x)` will call `f(x)` multiple times. Layers like `BatchNorm` with state may get confused.


### ReverseDiff.jl

* Like Tracker this passes a special TrackedArray type through your function. Allows you to record & compile the tape, and pre-allocate things.

* Needs a flat vector

* No support for GPU



<hr/>

## Second-order

If you calculate some `gradient(f, x)` inside the loss function, then `f` needs to be differentiated twice for the final result.

### Zygote over Zygote

In principle this works but in practice... best start small.

### ForwardDiff over Zygote

Zygote.hessian is like this.

### Enzyme.jl

I haven't tried really, but I think it ought to work.

<hr/>

## Meta-packages

Besides AD packages, several packages have been written aiming to provide a unified interface to many options. These may offer useful ways to quickly switch between things you are trying.

### [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl)

The original meta-package?

### [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl)

This year's new attempt to build a simpler one?

### [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl)

Really `rrule_via_ad` is another mechanism, but only for 3 systems.





