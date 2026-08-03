# Mixed Precision

Training in reduced floating point precision (`Float16` or `BFloat16`) can be
substantially faster on modern GPUs and halves the memory taken by activations.
Flux offers two complementary mechanisms, mirroring the two approaches available
in PyTorch:

1. **Autocast (recommended for training)**: the model's parameters stay in
   `Float32`, and a scoped [`autocast`](@ref) context casts values at layer-call
   time. This corresponds to PyTorch's `torch.autocast`.
2. **Static casting (recommended for inference)**: [`f16`](@ref) and
   [`bf16`](@ref) convert the parameters themselves, like PyTorch's
   `model.half()` and `model.bfloat16()`.

## Autocast

Wrap the forward pass — or simply pass the `autocast` keyword to
[`Flux.gradient`](@ref), [`Flux.withgradient`](@ref) or [`Flux.train!`](@ref):

```julia
using Flux

model = Chain(Conv((3, 3), 3 => 16, relu), BatchNorm(16),
              Flux.flatten, Dense(16 * 26 * 26 => 10)) |> gpu
opt_state = Flux.setup(Adam(1e-3), model)

for (x, y) in dataloader
    loss, grad = Flux.withgradient(model; autocast=BFloat16) do m
        Flux.logitcrossentropy(m(x), y)
    end
    Flux.update!(opt_state, model, grad[1])
end

# or, equivalently:
Flux.train!((m, x, y) -> Flux.logitcrossentropy(m(x), y), model, dataloader,
            opt_state; autocast=BFloat16)
```

Inside the scope:

- Matmul- and convolution-heavy layers (`Dense`, `Conv`, `ConvTranspose`,
  `CrossCor`, `Bilinear`, `Embedding` on onehot input, `MultiHeadAttention`, and
  the recurrent cells) cast their parameters and inputs to the requested half
  precision before computing, so the compute-intensive kernels run fast and the
  large activations take half the memory.
- Numerically sensitive operations compute in `Float32`: the normalization
  layers (`BatchNorm`, `LayerNorm`, `InstanceNorm`, `GroupNorm`) and the loss
  functions in `Flux.Losses` cast their inputs *up*.
- The parameters are never modified; they act as `Float32` "master weights".
  The backward pass of each cast accumulates the gradient back in `Float32`, so
  parameter gradients — and therefore the optimiser state and update — are
  full-precision, with no change to the training loop.

Things to keep in mind:

- **`Float16` can underflow.** Its narrow exponent range means small gradients
  flush to zero; robust `Float16` training typically needs dynamic loss scaling
  (PyTorch's `GradScaler`), which Flux does not provide yet. `BFloat16` has the
  same exponent range as `Float32` and trains reliably without it — prefer
  `BFloat16` where supported.
- Raw reductions in user code (e.g. `sum(abs2, m(x))`) are not intercepted: they
  compute in the half precision that flows into them, just as in PyTorch. Use
  the `Flux.Losses` functions, or cast to `Float32` yourself, for the final
  reduction.
- The softmax inside `MultiHeadAttention`'s `dot_product_attention` runs in the
  half precision (the projections produce half-precision `q`, `k`, `v`).
- Custom layers are not cast automatically unless they are built out of Flux
  layers. To opt in, consult [`Flux.autocast_eltype`](@ref) in your forward pass
  (see its docstring).
- Weights are re-cast on every forward pass (there is no cast cache). The cast
  is cheap next to the matmul/convolution it enables, and compiled backends fuse
  it away.
- Autocast is compiled out until its first use: models that never enter an
  `autocast` scope pay zero overhead and keep their exact inferred return types.
  The first `autocast` call in a session enables the machinery globally, which
  triggers a one-time recompilation of affected layer code.
- Autocast works with Zygote (the default), Mooncake, and — for `Float16` —
  Enzyme. `BFloat16` autocast is currently not supported with Enzyme.

## Static casting: `f16`, `bf16` and `f16mix`, `bf16mix`

For inference, converting the parameters once avoids the per-call casts:

```julia
model16 = bf16(model)      # full cast, like PyTorch's model.bfloat16()
model16 = bf16mix(model)   # same, but norm statistics/affine stay Float32
```

[`f16`](@ref)/[`bf16`](@ref) convert *every* parameter. [`f16mix`](@ref)/
[`bf16mix`](@ref) keep the statistics and affine parameters of `BatchNorm`,
`InstanceNorm` and `GroupNorm` in `Float32`, which the GPU normalization kernels
require for half-precision inputs — prefer the `mix` variants for models
containing those layers.

For *training* a statically converted model, the gradients and optimiser state
are also half-precision, which loses update accuracy. The
[`Optimisers.MixedPrecision`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.MixedPrecision) rule
compensates by keeping a `Float32` copy of the parameters inside the optimiser
state:

```julia
model16 = bf16mix(model)
opt_state = Flux.setup(Optimisers.MixedPrecision(Adam(1e-3)), model16)
```

Compared to autocast this halves the model's parameter memory (at the price of
the extra copy in the optimiser state) but computes *everything* except the norm
layers in half precision, including the numerically sensitive reductions.

## Custom layers under autocast

Flux's built-in layers consult the ambient autocast scope in their forward pass.
A custom layer written in terms of Flux layers (e.g. a struct holding a `Dense`)
inherits this for free. A layer that multiplies its own weight arrays needs one
extra line to participate:

```julia
struct Affine{W, B}
    weight::W
    bias::B
end
Flux.@layer Affine

function (a::Affine)(x)
    Flux._autocast_barrier() do T   # T is Float16, BFloat16, or nothing
        W = Flux._autocast_down(T, a.weight)
        b = Flux._autocast_down(T, a.bias)
        xT = Flux._autocast_down(T, x)
        return W * xT .+ b
    end
end
```

`Flux._autocast_down(T, x)` casts a floating-point array to `T` and is a no-op
when `T === nothing` (no active scope), when the array already has eltype `T`,
and for non-float arrays (integer or onehot inputs, a `false` bias, ...); its
gradient casts back, so parameter gradients stay `Float32`. The
`Flux._autocast_barrier` wrapper runs the closure with the current scope value
and acts as a dispatch barrier, keeping the layer type-stable when autocast is
not in use. For a numerically sensitive custom layer, use `Flux._autocast_up(x)`
instead, which casts half-precision input *up* to `Float32` inside a scope.

!!! warning
    Inside the `_autocast_barrier` closure, only assign to *fresh* local names.
    Assigning to a variable captured from the enclosing function (like `x` or a
    destructured argument) boxes it, which silently breaks both type inference
    and Zygote gradients.

These helpers are currently internal (underscore-prefixed): the API may still
evolve, but they are the supported way to make a custom layer autocast-aware.

Note that when writing a custom layer as a plain Julia function of arrays, an
alternative is to rely on [`Flux.autocast_eltype`](@ref) directly.
