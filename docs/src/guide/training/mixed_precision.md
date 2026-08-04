# Mixed Precision

Training in reduced floating point precision (`Float16` or `BFloat16`) can be
substantially faster on modern GPUs and halves the memory taken by activations.
Flux offers two complementary mechanisms, mirroring the two approaches available
in PyTorch:

1. **Autocast (recommended for training)**: the model's parameters stay in
   `Float32`, and [`autocast`](@ref) wraps the model so that layers cast at
   call time. This corresponds to PyTorch's `torch.autocast`.
2. **Static casting (recommended for inference)**: [`f16`](@ref) and
   [`bf16`](@ref) convert the parameters themselves, like PyTorch's
   `model.half()` and `model.bfloat16()`.

## Autocast

Pass the `autocast` keyword to [`Flux.gradient`](@ref), [`Flux.withgradient`](@ref)
or [`Flux.train!`](@ref):

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

`autocast(model, T)` returns a *wrapped* model that shares `model`'s parameter
arrays (used automatically by the keyword above; you can also build it yourself
for inference: `autocast(model, BFloat16)(x)`). In the wrapped model:

- Matmul- and convolution-heavy layers (`Dense`, `Conv`, `ConvTranspose`,
  `CrossCor`, `Bilinear`, `MultiHeadAttention`'s projections, and the recurrent
  cells) cast their parameters and inputs to the requested half precision before
  computing, so the compute-intensive kernels run fast and the large activations
  take half the memory.
- Numerically sensitive operations keep their statistics in `Float32`.
  `BatchNorm` folds its statistics in `Float32` inside the kernel (natively on the
  GPU, via cuDNN) while letting the half-precision activation pass straight
  through, so it neither loses accuracy nor pays a full-precision round-trip; on
  the CPU, where there is no such kernel (and half precision is not faster), it
  casts up to `Float32` instead. `LayerNorm`, `InstanceNorm` and `GroupNorm`
  always cast their input *up* to `Float32`. The loss functions in `Flux.Losses`
  always accumulate in `Float32` when given half-precision inputs.
- The parameters are never modified; they act as `Float32` "master weights".
  The backward pass of each cast accumulates the gradient back in `Float32`, and
  the gradient returned by `gradient`/`withgradient` is shaped like the *original*
  (unwrapped) model — so the optimiser state and update need no change.

Because the wrapping is an ordinary (differentiable) model transformation rather
than a runtime scope, both plain and wrapped forward passes keep their **exact
inferred element type** — there is no type-inference penalty for defining
`autocast` and no cost for models that never use it.

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
- `Embedding` is left in full precision (casting the — often large — embedding
  table on every forward would be expensive, and PyTorch keeps embeddings in
  `Float32` under autocast); downstream wrapped layers cast the looked-up vectors.
- Parameters are re-cast on every forward pass (there is no cast cache). The cast
  is cheap next to the matmul/convolution it enables, and compiled backends fuse
  it away.
- Autocast works with Zygote (the default), Mooncake, and — for `Float16` —
  Enzyme. `BFloat16` autocast is currently not supported with Enzyme
  ([EnzymeAD/Enzyme.jl#3430](https://github.com/EnzymeAD/Enzyme.jl/issues/3430)).

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

A custom layer built out of Flux layers (e.g. a struct holding a `Dense`) is
handled automatically: `autocast` recurses into it and wraps the inner layers. A
layer that multiplies its *own* weight arrays opts in with a one-line trait,
[`Flux.autocast_mode`](@ref):

```julia
struct Affine{W, B}
    weight::W
    bias::B
end
Flux.@layer Affine

(a::Affine)(x) = a.weight * x .+ a.bias

Flux.autocast_mode(::Affine) = :down   # cast this layer's params + inputs to half precision
```

`autocast` then wraps `Affine` like a built-in `Dense`: on each call its
floating-point parameters and inputs are cast to the half-precision type (its
gradients still come back in `Float32`), and the wrapped forward pass keeps its
exact inferred element type. Return `:up` instead for a numerically sensitive
layer that should compute in `Float32` (like the normalization layers), or the
default `:none` to leave a layer untouched and let `autocast` recurse into it.
