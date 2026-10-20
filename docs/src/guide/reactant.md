# Compiling Flux with Reactant

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) traces your Julia code and compiles it — through [MLIR](https://mlir.llvm.org/) and [XLA](https://openxla.org/xla) — into a single optimised executable that runs on CPU, NVIDIA/AMD GPUs, or TPUs. For Flux this means the whole forward pass (and, for training, the [Enzyme](https://enzyme.mit.edu/) reverse pass and the optimiser update) is fused, kernel-optimised, and reused across calls, which is often substantially faster and more memory-efficient than eager execution.

This guide builds up in three steps:

1. A **manually compiled** example, so you can see exactly what Reactant does.
2. The [`trainstep!`](@ref Flux.Train.trainstep!) API, which compiles and caches a single training step for you.
3. [`train!`](@ref Flux.train!), the full training loop, which is built on top of `trainstep!`.

## Installation

Reactant is a normal dependency — add it and load it alongside Flux:

```julia
using Pkg; Pkg.add("Reactant")  # do this once

using Flux, Reactant
```

Reactant automatically selects a GPU backend if one is available, falling back to the CPU otherwise. See the [Reactant GPU configuration docs](https://enzymead.github.io/Reactant.jl/dev/api/config#GPU-Configuration) for how to control this.

## Moving data and models to the device

Just like `gpu`/`cpu`, Flux provides a device object that moves a model (or any nested structure of arrays) onto the Reactant device, converting its arrays into Reactant's `ConcreteRArray`s:

```julia
using Flux, Reactant

dev = reactant_device()   # a Reactant device object; call once and reuse

model = Chain(Dense(4 => 8, tanh), Dense(8 => 2))
x = randn(Float32, 4, 16)

model_re = model |> dev   # a copy of the model with Reactant arrays
x_re     = x     |> dev
```

As with the GPU adaptors, `dev` recurses into structures, so `(x, y) |> dev` moves both. It keeps the element type, so `Float64` arrays stay `Float64`: call [`f32`](@ref Flux.f32) first, or use `reactant_device(Float32)`, to convert them. Everything you feed to a compiled function must already live on the device.

## 1. A manually compiled example

Reactant does not run your code eagerly. Instead you *compile* a function once for a given set of input shapes and types, and then call the returned executable. The two entry points are:

- `Reactant.@compile f(args...)` — trace and compile `f`, returning a callable executable. Call it later with device-resident arguments of the same shapes.
- `Reactant.@jit f(args...)` — compile *and* immediately run, a convenient shortcut for one-off calls.

### Compiling the forward pass (inference)

```julia
using Flux, Reactant

dev = reactant_device()
model = Chain(Dense(4 => 8, tanh), Dense(8 => 2)) |> dev
x = randn(Float32, 4, 16) |> dev

# Compile once...
forward = Reactant.@compile model(x)

# ...then call the compiled executable (fast, no recompilation):
y = forward(x)              # a Reactant array on the device
y_host = y |> cpu           # move the result back to the host

# `@jit` compiles and runs in one go — handy for a one-off evaluation:
y2 = Reactant.@jit model(x)
```

Note the call is `forward(x)`, **not** `forward(model, x)`: in `@compile model(x)` the model is the *callee*, so Reactant captures it (and its parameter arrays) inside the compiled executable, and you pass only the remaining arguments. Because the model is captured by reference, mutating its parameters in place — as training does — is reflected the next time you call `forward`. (Contrast this with the training-step examples below, where the model sits in an *argument* position, e.g. `@compile my_step!(loss, model, x, y, opt_state)`, and so must be passed at call time.)

A compiled executable is specialised to the **shapes** of its inputs. Calling `forward` with a differently-shaped `x` (e.g. a smaller final batch) requires compiling a separate executable for that shape.

!!! note "trainmode / testmode"
    Layers such as `Dropout` and `BatchNorm` behave differently during training and inference. Call `testmode!(model)` before compiling an inference function, and `trainmode!(model)` before a training step, exactly as you would without Reactant.

### Compiling a training step manually

To differentiate under Reactant, use Flux's [`withgradient`](@ref Flux.withgradient) with the [`AutoEnzyme`](@ref) backend — Reactant relies on Enzyme for AD. You can compile the value-and-gradient computation, or a whole step that also updates the model in place:

```julia
using Flux, Reactant, Optimisers

dev = reactant_device()
model = Chain(Dense(4 => 8, tanh), Dense(8 => 2)) |> dev
x, y = randn(Float32, 4, 16) |> dev, randn(Float32, 2, 16) |> dev

loss(m, x, y) = Flux.mse(m(x), y)
opt_state = Flux.setup(Adam(1f-2), model)

# A plain Julia function describing one optimisation step. It differentiates the loss
# with Enzyme, updates the model and optimiser state in place, and returns the loss.
function my_step!(loss, model, x, y, opt_state)
    l, grads = Flux.withgradient(m -> loss(m, x, y), AutoEnzyme(), model)
    Optimisers.update!(opt_state, model, grads[1])
    return l
end

trainmode!(model)
step! = Reactant.@compile my_step!(loss, model, x, y, opt_state)   # compile once

for epoch in 1:100
    l = step!(loss, model, x, y, opt_state)   # reuse the executable each epoch
    @info "epoch $epoch" loss=Reactant.to_number(l)
end
```

Reactant traces the mutation of `model` and `opt_state` and fuses the whole step. Writing this by hand gives you full control, but it is boilerplate that Flux can handle for you — which is what the next section is about.

## 2. The `trainstep!` API

[`trainstep!`](@ref Flux.Train.trainstep!) performs exactly the step above — differentiate the loss, update `model` and `opt_state` in place, return the loss — and when the model lives on a Reactant device it **compiles and caches the fused step automatically**. You do not write `@compile` yourself, and repeated calls with the same model, optimiser, loss and batch shape reuse the cached executable.

```julia
using Flux, Reactant

dev = reactant_device()
model = Chain(Dense(4 => 8, tanh), Dense(8 => 2)) |> dev
x, y = randn(Float32, 4, 16) |> dev, randn(Float32, 2, 16) |> dev

loss(m, x, y) = Flux.mse(m(x), y)

# Move the model to the device *before* `setup`, so the optimiser is set up for Reactant arrays.
opt_state = Flux.setup(Adam(1f-2), model)

trainmode!(model)
for epoch in 1:100
    l = Flux.trainstep!(loss, model, (x, y), opt_state)   # returns the host-side loss scalar
    @info "epoch $epoch" loss=l
end
```

The batch is passed as a tuple `(x, y)` and spliced into the loss as `loss(model, x, y)`. The returned loss is read back to the host as an ordinary number.

Use [`trainstep_withgradient!`](@ref Flux.Train.trainstep_withgradient!) if you also need the gradient — it returns `(loss, grad)`, with the gradient left on the device. (Returning the gradient makes it an output of the compiled step and raises peak memory, so prefer `trainstep!` when you don't need it.)

### Auxiliary loss outputs

Like `withgradient`, the loss may return auxiliary data alongside the scalar loss: return a `Tuple` or `NamedTuple` whose first element is the loss, and the gradient is taken of the loss alone while the whole value is returned. This is convenient for logging a metric computed in the forward pass — and, on Reactant, the metric is computed on-device as part of the same compiled forward:

```julia
function loss(m, x, y)
    ŷ = m(x)
    Flux.mse(ŷ, y), (; acc = mean(onecold(ŷ) .== onecold(y)))
end

l, stats = Flux.trainstep!(loss, model, (x, y), opt_state)   # l is the loss, stats.acc the accuracy, both on the host
```

## 3. The `train!` API

[`train!`](@ref Flux.train!) is a loop over the data built on top of `trainstep!`, so on a Reactant device it inherits the same automatic compile-and-cache. Move the model to the device and call `setup` first, and provide **device-resident** data — `train!` rejects host arrays on the Reactant path.

```julia
using Flux, Reactant

dev = reactant_device()
model = Chain(Dense(4 => 8, tanh), Dense(8 => 2)) |> dev
opt_state = Flux.setup(Adam(1f-2), model)

loss(m, x, y) = Flux.mse(m(x), y)

# Move every batch to the device. In a real loop use a DataLoader wrapped with the device,
# so each batch is moved lazily and the previous one is freed:
#   train_loader = DataLoader((X, Y); batchsize=32) |> dev
X, Y = randn(Float32, 4, 128), randn(Float32, 2, 128)
data = [(X[:, i:i+15], Y[:, i:i+15]) |> dev for i in 1:16:128]

Flux.train!(loss, model, data, opt_state)
```

`train!` runs a single step per batch, shows a progress bar, and stops with a `DomainError` if the loss becomes non-finite. As in eager mode, the compiled step skips the update when the loss is non-finite, so the parameters and optimiser state keep their values from before that batch. Because the compiled step is cached and keyed on the batch shape, multi-epoch training does not recompile, and a smaller final batch simply compiles one additional executable that is then reused.

### Resuming training from a checkpoint

An optimiser state saved on the host (see [Checkpointing](@ref)) can't be moved to the device with `|> dev`. Instead, set up a new state for the model on the device, then load the saved one into it with [`loadmodel!`](@ref Flux.loadmodel!):

```julia
using JLD2
ckpt = JLD2.load("checkpoint.jld2")   # saved with jldsave("checkpoint.jld2"; model_state, opt_state)

model = Flux.loadmodel!(MyModel(), ckpt["model_state"]) |> dev
opt_state = Flux.setup(Adam(1f-2), model)
Flux.loadmodel!(opt_state, ckpt["opt_state"])
```

`loadmodel!` copies the optimiser's running state, such as `Adam`'s moments and bias-correction terms, but keeps the rule from `setup`, whose hyperparameters live on the device so that [`adjust!`](@ref Flux.adjust!) keeps working. To save a checkpoint from the device, move the states to the host first: `jldsave("checkpoint.jld2"; model_state = cpu(Flux.state(model)), opt_state = cpu(Flux.state(opt_state)))`.

## Tips and gotchas

- **`setup` after moving to the device.** Always `model |> reactant_device()` *before* `Flux.setup`, so the optimiser state is created from Reactant arrays (this keeps stateful rules like `Adam`'s bias-correction term on the device). `trainstep!` and `train!` reject a state set up on the host; to resume from a saved one, see [Resuming training from a checkpoint](@ref).
- **Device-resident data.** Every input to a compiled step must already live on the device. Move batches with `|> reactant_device()`; a common pattern is to wrap a `DataLoader` with the device so batches are moved lazily.
- **Scalars in the batch.** A plain number in the batch, such as `λ` in `(x, y, λ)`, is compiled into the step as a constant, so each distinct value compiles a new step. Pass a varying scalar as a 0-dimensional array instead, `(x, y, fill(λ)) |> dev`, and read it with `λ[]` in the loss.
- **Compilation is keyed on shape.** The first call for each distinct batch shape compiles; subsequent calls reuse the executable. Keeping batch sizes fixed (or accepting one extra compile for a smaller final batch) avoids repeated compilation.
- **Watch the compile cache.** Flux caches one executable per distinct `(model, optimiser, loss, batch-shape)`. Values that are compiled into the step as constants are part of the cache key too, so switching a layer with `trainmode!`/`testmode!`, freezing parameters with `freeze!`/`thaw!`, passing the same array twice in a batch, or redefining the loss function compiles a new step instead of reusing a stale one. If you rebuild the model or optimiser every iteration, each iteration compiles a fresh step — Flux warns when the cache grows past a handful of entries. Entries are freed automatically when their model is garbage-collected.
- **Reading results back.** `trainstep!`/`train!` return host-side numbers already. For a manually compiled function, move device arrays back with `cpu` (or `Reactant.to_number` for a scalar).
- **AD backend.** On a Reactant device training differentiates with Enzyme, the only supported backend: passing any other `adtype`, such as `AutoZygote()` or `AutoMooncake()`, throws an `ArgumentError`. `Duplicated` models are not supported here — pass the plain model that already lives on the device.

!!! tip "Precision on GPU"
    On CUDA, Reactant's default precision for `Float32` matrix multiplications and convolutions behaves like TF32, trading accuracy for speed. On an RTX 5090 the gradient of a small MLP differed from the `Float64` one by 6e-4 (relative error), and that of a small CNN by 3e-3, against under 4e-7 at full precision. To get full `Float32` precision, compile the step inside `Reactant.with_config`:

    ```julia
    Reactant.with_config(; dot_general_precision = PrecisionConfig.HIGHEST,
                           convolution_precision = PrecisionConfig.HIGHEST) do
        Flux.trainstep!(loss, model, (x, y), opt_state)
    end
    ```

    The setting is read when the step is compiled, i.e. on the first `trainstep!` call for a given model, optimiser, loss and batch shape, so wrap that call (or the whole training loop). A step that is already compiled keeps its precision: calling it inside `with_config` later has no effect, and a step compiled with `HIGHEST` keeps it outside.

See also the [`resnet_tinyimagenet` example](https://github.com/FluxML/Flux.jl/tree/master/examples/resnet_tinyimagenet) for a complete Reactant training script, including compiling a separate evaluation executable per batch shape, and the [`gpt2_fineweb` example](https://github.com/FluxML/Flux.jl/tree/master/examples/gpt2_fineweb), which pretrains GPT-2 with gradient accumulation, bfloat16 matrix multiplications and checkpointing.
