# Compiling Flux with Reactant

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) compiles Julia code into XLA executables for CPU, supported GPUs and TPUs. On a Reactant device, Flux's [`train!`](@ref Flux.train!) and [`trainstep!`](@ref Flux.Train.trainstep!) compile the forward pass, gradient calculation with Enzyme and optimiser update together.

## Installation

Install Reactant alongside Flux by typing `] add Reactant` at the Julia prompt.

By default, Reactant selects a supported GPU if one is available and otherwise uses the CPU. See the [Reactant GPU configuration docs](https://enzymead.github.io/Reactant.jl/dev/api/config#GPU-Configuration) to control this.

## [Train with `train!`](@id 3.-The-train!-API)

Move the model to the Reactant device before calling `Flux.setup`. Wrap the data loader with the same device to move each batch as it is loaded.

```julia
using Flux, Reactant

dev = reactant_device()
model = Chain(Dense(4 => 8, tanh), Dense(8 => 2)) |> dev
opt_state = Flux.setup(Adam(1f-2), model)

X, Y = randn(Float32, 4, 128), randn(Float32, 2, 128)
loader = Flux.DataLoader((X, Y); batchsize=16) |> dev
loss(m, x, y) = Flux.mse(m(x), y)

for _ in 1:3
    Flux.train!(loss, model, loader, opt_state)
end
```

The first batch triggers compilation. Later batches of the same shape reuse the executable, including across epochs. A different shape, such as a smaller final batch, triggers another compilation.

`train!` shows a progress bar and throws a `DomainError` for a non-finite loss. The compiled step skips the update for that batch. Flux checks each loss after queuing the following step, so that step can still update the model.

## Moving data and models to the device

`reactant_device()` returns a device object that converts arrays, including those inside models and tuples, to Reactant arrays. For example, `(x, y) |> dev` moves both arrays. The device preserves element types by default. Use `reactant_device(Float32)` or call [`f32`](@ref Flux.f32) before moving a model to convert floating-point arrays to `Float32`.

## [Train one batch with `trainstep!`](@id 2.-The-trainstep!-API)

Use [`trainstep!`](@ref Flux.Train.trainstep!) to write your own training loop. Each call evaluates the loss, computes the gradient and updates the model and optimiser state in place. On a Reactant device, Flux compiles and caches the step automatically.

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
    l = Flux.trainstep!(loss, model, (x, y), opt_state)   # the loss, on the device
    @info "epoch $epoch" loss=Float32(l)
end
```

The batch is passed as a tuple `(x, y)` and spliced into the loss as `loss(model, x, y)`. The returned loss stays on the device as a Reactant number: reading it, as `Float32(l)` does here, waits for the step to finish (see [Reading the loss without waiting](@ref)).

Use [`trainstep_withgradient!`](@ref Flux.Train.trainstep_withgradient!) if you also need the gradient — it returns `(loss, grad)`, with the gradient left on the device. (Returning the gradient makes it an output of the compiled step and raises peak memory, so prefer `trainstep!` when you don't need it.)

### Auxiliary loss outputs

Like `withgradient`, the loss may return auxiliary data alongside the scalar loss: return a `Tuple` or `NamedTuple` whose first element is the loss, and the gradient is taken of the loss alone while the whole value is returned. This is convenient for logging a metric computed in the forward pass — and, on Reactant, the metric is computed on-device as part of the same compiled forward:

```julia
function loss(m, x, y)
    ŷ = m(x)
    Flux.mse(ŷ, y), (; acc = mean(onecold(ŷ) .== onecold(y)))
end

l, stats = Flux.trainstep!(loss, model, (x, y), opt_state)   # l is the loss, stats.acc the accuracy, both on the device
```

### Reading the loss without waiting

`trainstep!` returns as soon as the step is queued on the device, and the loss it returns stays there. Reading it — with `Float32(l)`, arithmetic such as `total += l`, `isfinite(l)`, or printing — waits for the step to finish. A loop that reads every loss therefore waits for each step before it prepares the next batch, while the device sits idle. Read the loss only every `log_interval` steps instead, so that the device runs the steps in between while the host prepares their batches:

```julia
train_loader = DataLoader((X, Y); batchsize=32) |> dev
log_interval = 10
for (i, (x, y)) in enumerate(train_loader)
    l = Flux.trainstep!(loss, model, (x, y), opt_state)
    i % log_interval == 0 && @info "step $i" loss=Float32(l)   # waits for this step
end
```

This helps most when preparing a batch takes a good fraction of the time of a step, for instance when loading and augmenting images in the training loop itself. To make the preparation itself faster, let the [`DataLoader`](@ref Flux.DataLoader) load batches in parallel: `parallel = true` uses worker threads (start Julia with several threads), and `num_workers = N` uses `N` worker processes (MLUtils 0.4.10 and later), which also speeds up loading that threads can't run in parallel, such as from a Python-backed dataset. The two combine: the workers prepare the batches, while what remains on the main thread, such as moving each batch to the device, overlaps with the device's steps. Between two reads the host can queue up to `log_interval` steps, and the device keeps the batch of each until it runs it. [`train!`](@ref Flux.train!) checks each loss once the next step is queued, so its check doesn't keep the device waiting.

## [Manual compilation](@id 1.-A-manually-compiled-example)

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

To differentiate under Reactant, call Flux's [`withgradient`](@ref Flux.withgradient) with `AutoEnzyme()`. You can compile the value and gradient or include the in-place optimiser update in the same executable. See [automatic differentiation with Enzyme](@ref autodiff-enzyme).

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

Reactant traces the updates to `model` and `opt_state`. It compiles them together with the forward pass and the gradient calculation.

## Resuming training from a checkpoint

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
- **Reading results back.** `trainstep!` returns the loss on the device: read it with `Float32(l)` or `Reactant.to_number(l)`, preferably only every few steps (see [Reading the loss without waiting](@ref)). Move device arrays back with `cpu`, as for the results of a manually compiled function.
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

## End-to-end examples

For complete training scripts, see [ResNet on TinyImageNet](https://github.com/FluxML/Flux.jl/tree/master/examples/resnet_tinyimagenet) and [GPT-2 pretraining](https://github.com/FluxML/Flux.jl/tree/master/examples/gpt2_fineweb). The ResNet example uses fixed-size training batches and compiles evaluation separately. The GPT-2 example uses gradient accumulation, mixed precision and checkpointing.
