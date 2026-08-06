# [Performance Tips](@id man-performance-tips)

All the usual [Julia performance tips apply](https://docs.julialang.org/en/v1/manual/performance-tips/).
As always [profiling your code](https://docs.julialang.org/en/v1/manual/profile/#Profiling-1) is generally a useful way of finding bottlenecks.
Below follow some Flux specific tips/reminders.

## Don't use more precision than you need

Flux works great with all kinds of number types.
But often you do not need to be working with say `Float64` (let alone `BigFloat`).
Switching to `Float32` can give you a significant speed up,
not because the operations are faster, but because the memory usage is halved.
Which means allocations occur much faster.
And you use less memory.

## Preserve inputs' types

Not only should your activation and loss functions be [type-stable](https://docs.julialang.org/en/v1/manual/performance-tips/#Write-%22type-stable%22-functions-1),
they should also preserve the type of their inputs.

A very artificial example using an activation function like

```julia
my_tanh(x) = Float64(tanh(x))
```

will result in performance on `Float32` input orders of magnitude slower than the normal `tanh` would,
because it results in having to use slow mixed type multiplication in the dense layers.
Similar situations can occur in the loss function during backpropagation.

Which means if you change your data say from `Float64` to `Float32` (which should give a speedup: see above),
you will see a large slow-down.

This can occur sneakily, because you can cause type-promotion by interacting with a numeric literals.
E.g. the following will have run into the same problem as above:

```julia
leaky_tanh(x) = 0.01*x + tanh(x)
```

While one could change the activation function (e.g. to use `0.01f0*x`), the idiomatic (and safe way)  to avoid type casts whenever inputs changes is to use `oftype`:

```julia
leaky_tanh(x) = oftype(x/1, 0.01)*x + tanh(x)
```

## Evaluate batches as matrices of features

While it can sometimes be tempting to process your observations (feature vectors) one at a time
e.g.

```julia
function loss_total(xs::AbstractVector{<:Vector}, ys::AbstractVector{<:Vector})
    sum(zip(xs, ys)) do (x, y_target)
        y_pred = model(x)  # evaluate the model
        return loss(y_pred, y_target)
    end
end
```

It is much faster to concatenate them into a matrix,
as this will hit BLAS matrix-matrix multiplication, which is much faster than the equivalent sequence of matrix-vector multiplications.
The improvement is enough that it is worthwhile allocating new memory to store them contiguously.

```julia
x_batch = reduce(hcat, xs)
y_batch = reduce(hcat, ys)
...
function loss_total(x_batch::Matrix, y_batch::Matrix)
    y_preds = model(x_batch)
    sum(loss.(y_preds, y_batch))
end
```

When doing this kind of concatenation use `reduce(hcat, xs)` rather than `hcat(xs...)`.
This will avoid the splatting penalty, and will hit the optimised `reduce` method.

## Be aware of GPU memory inefficiencies

Currently, GPU memory is not handled as well as system memory. If your training loop is allocating significantly on the GPU, you can quickly fill your GPU memory and the piecemeal reclamation and shuffling of data between GPU and system memory can become extremely slow. If profiling shows that a significant portion of time is spent in the `gpu` function and your data sizes are not large, this may be the cause. Running an incremental garbage collection manually (`GC.gc(false)`) at regular intervals can keep your GPU memory free and responsive. See other tips for CUDA memory management [here](https://cuda.juliagpu.org/stable/usage/memory/).

## Compile your training with Reactant

The tips above tune *eager* execution. For the largest speedups, compile your model with [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl): it traces your code and lowers it — through MLIR and XLA — into a single fused, kernel-optimised executable that is cached and reused across calls. For training, the forward pass, the Enzyme reverse pass and the optimiser update are compiled *together*, which removes per-operation dispatch and kernel-launch overhead and lets the compiler fuse operations and plan buffers across the whole step.

You do not have to invoke the compiler yourself: when the model lives on a Reactant device, [`trainstep!`](@ref Flux.Train.trainstep!) — and [`train!`](@ref Flux.train!), which is built on it — compile and cache the fused step automatically.

```julia
using Flux, Reactant

dev = reactant_device()
model = model |> dev
opt_state = Flux.setup(Adam(1f-3), model)   # set up the optimiser after moving to the device
loader = DataLoader((X, Y); batchsize=32) |> dev  # move batches to the device during iteration
loss(m, x, y) = Flux.logitcrossentropy(m(x), y)
trainmode!(model)
for (x, y) in loader
    Flux.trainstep!(loss, model, (x, y), opt_state)   # compiled on the first call, reused after
end
```

The cached executable is keyed on the batch shape, so keeping batch sizes fixed avoids recompilation (a smaller final batch simply compiles one extra executable). The first call pays a one-time compilation cost that can be large, so this pays off over many steps rather than a handful. See [Compiling Flux with Reactant](reactant.md) for the full guide, including inference and manual `@compile`.
