# [Mixed Precision Training](@id man-mixed-precision)

Training with 16-bit floating point numbers halves the memory taken by parameters and
activations, and on GPUs with 16-bit tensor cores it can make training faster. Flux has the
pieces for this: [`bf16`](@ref Flux.bf16) and [`f16`](@ref Flux.f16) convert a model, and
[`MixedPrecision`](@ref Optimisers.MixedPrecision) from Optimisers.jl keeps the updates
accurate. This page shows how to combine them.

## BFloat16 or Float16?

Both use 16 bits, split differently:

* `BFloat16` keeps the 8 exponent bits of `Float32`, and so its range (up to about `3.4e38`),
  but has only 8 significant bits.
* `Float16` has 11 significant bits, but its largest value is `65504`, and small gradients
  can underflow to zero.

Training in `Float16` usually needs *loss scaling* (multiplying the loss by a large factor
and dividing the gradients by it) to avoid the underflow. Flux does not do loss scaling, so
**use `BFloat16`**. The `BFloat16` type comes from
[BFloat16s.jl](https://github.com/JuliaMath/BFloat16s.jl).

16-bit arithmetic is a GPU feature. Most CPUs have no 16-bit arithmetic, so Julia converts
to `Float32` for each operation, which is slower than staying in `Float32`.

## The recipe

Convert the model with [`bf16`](@ref Flux.bf16), and wrap the optimisation rule in
[`MixedPrecision`](@ref Optimisers.MixedPrecision):

```julia
using Flux, BFloat16s

model = Chain(Dense(784 => 256, relu), BatchNorm(256), Dense(256 => 10)) |> bf16
opt_state = Flux.setup(MixedPrecision(Adam(1e-3)), model)

x = rand(Float32, 784, 64) |> bf16
y = Flux.onehotbatch(rand(0:9, 64), 0:9)

grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
Flux.update!(opt_state, model, grads[1])
```

The forward and backward passes then run in `BFloat16`, and so do the gradients.

The normalisation layers `BatchNorm`, `InstanceNorm`, `GroupNorm` and `LayerNorm` are not
converted: their statistics and affine parameters stay `Float32` while the data flowing
through them is `BFloat16`, as cuDNN requires.

`MixedPrecision` keeps a `Float32` copy of each parameter, and the state of the wrapped rule
(here Adam's moments) in `Float32` too. Each step applies the update to the `Float32` copy
and then writes its `BFloat16` rounding into the model. Without it, an update smaller than
the spacing between `BFloat16` numbers near a weight would be rounded away.

There is another way, from [#2291](https://github.com/FluxML/Flux.jl/issues/2291): keep the
model in `Float32`, and differentiate a `bf16` copy of it,

```julia
model32 = Chain(Dense(784 => 256, relu), Dense(256 => 10))
opt_state = Flux.setup(Adam(1e-3), model32)

grads = Flux.gradient(m -> Flux.logitcrossentropy(bf16(m)(x), y), model32)
Flux.update!(opt_state, model32, grads[1])
```

This converts the model at every step, but the model you train is the `Float32` one, and
`opt_state` holds no extra copy. Prefer `MixedPrecision`, which converts nothing per step;
use the second form when you want the `Float32` model at hand, for example to evaluate or
save it during training.

## Pitfalls

**Don't train `BFloat16` parameters without `MixedPrecision`.** Plain `Adam` on `BFloat16`
parameters gives `NaN` on the first step, because `β₂ = 0.999` rounds to `1.0` in
`BFloat16` and Adam divides by `1 - β₂^t`
([Optimisers.jl#215](https://github.com/FluxML/Optimisers.jl/issues/215)). Other rules lose
small updates for the reason above. Always wrap the rule:

```julia
opt_state = Flux.setup(MixedPrecision(Adam(1e-3)), model)  # not Flux.setup(Adam(1e-3), model)
```

**Save `opt_state` along with the model.** With `MixedPrecision` the weights being trained
are the `Float32` copies inside `opt_state`; `Flux.state(model)` holds only their `BFloat16`
rounding. To resume training exactly, save both, as described in
[Saving & Loading](saving.md).

**Use current versions.** Older releases computed some losses and `softmax`/`logsoftmax`/
`logsumexp` in the precision of their input, which over many classes gives very wrong
results in 16 bits, and under Zygote a `Float32` operation on `BFloat16` values made the
gradients `Float32`. These are fixed in Flux 0.16.15
([#2757](https://github.com/FluxML/Flux.jl/issues/2757)), NNlib 0.9.59
([NNlib.jl#813](https://github.com/FluxML/NNlib.jl/issues/813)) and ChainRulesCore 1.26.4
([ChainRulesCore.jl#718](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/718)).

## What to expect

How much faster training gets depends on the GPU and the model. For example, on an
RTX 5090, ResNet-18 with batch size 128 trained 1.31× faster with `bf16` and `MixedPrecision`
than in `Float32`, and used 42% less peak memory. Measure your own model, as in
[Profiling a Training Loop](@ref man-profiling), before relying on a speedup.
