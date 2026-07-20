# VGG16 / CIFAR-10 — a perf playground for `train_step!`

A runnable example, adapted from the
[model-zoo VGG/CIFAR-10 script](https://github.com/FluxML/model-zoo/blob/master/vision/vgg_cifar10/vgg_cifar10.jl),
used to measure the performance (time per epoch and GPU memory) of the step-by-step
training API added in [PR #2681](https://github.com/FluxML/Flux.jl/pull/2681)
(`Flux.TrainState` + `Flux.train_step!`).

It contrasts two ways of taking an optimisation step:

* `:train_step` — `Flux.TrainState` + `Flux.train_step!`, which wraps each step in a
  `GPUArrays.@cached` block backed by a `GPUArrays.AllocCache` reused across steps.
* `:manual` — the classic `gradient` + `update!` loop (the model-zoo baseline).

## Running

```julia
julia --project

julia> include("vgg_cifar10.jl")

julia> main(; method = :train_step, batchsize = 128, epochs = 3, ntrain = 5120)
julia> main(; method = :manual,     batchsize = 128, epochs = 3, ntrain = 5120)
```

`ntrain = 0` uses the full 50k training set; a smaller value is handy for quick runs.
`bench_driver.jl` runs a single config from the command line so each run gets a **fresh
GPU process** (important — the memory pool and the `AllocCache` do not shrink, so
comparing methods in the same process is misleading):

```bash
julia --project bench_driver.jl train_step 128 3 5120
julia --project bench_driver.jl manual     128 3 5120
```

## What is measured

* per-epoch wall-clock time (with `CUDA.synchronize()` around the epoch);
* `peak_used` — max of `CUDA.used_memory()` (live pool bytes) sampled after each step;
* `reserved`  — `used + cached` bytes the pool is holding.

The first epoch includes compilation; look at epochs ≥ 2 for steady-state timing.

## Findings (RTX 5090, 32 GiB; 5120 imgs, 3 epochs; each config a fresh process)

Peak *live* GPU memory (`CUDA.used_memory()` sampled per step) and what the `AllocCache`
ends up holding, varying only where `@cached` wraps the step:

| method         | bs128 peak | bs128 cache | bs32 peak | bs32 cache |
|----------------|-----------:|------------:|----------:|-----------:|
| `:manual`      |   17592    |      0      |   17121   |      0     |
| `:cache_both`  |   26767    |    25155    |    8292   |    6717    |
| `:cache_grad`  |   26782    |    25175    |    8285   |    6713    |
| `:cache_update`|   18452    |      0      |   18676   |      0     |

(`:cache_both` is what `Flux.train_step!` does today.) Steady-state epoch time was identical
across all methods — the cache changes memory, not speed.

### The whole cost is the gradient pass; splitting the step does not help

* `:cache_grad` ≈ `:cache_both`: the cache holds ~25 GiB (bs128) / ~6.7 GiB (bs32) either
  way. **The entire cached footprint is the gradient pass** — the forward activations kept
  for the backward pass.
* `:cache_update` ≈ `:manual`, cache holds **0 MiB**: `update!` (in-place Adam) allocates
  essentially nothing cacheable.

So putting the gradient and the `update!` in *separate* `@cached` blocks changes nothing:
you either cache the gradient (and pay the full footprint) or you don't.

### Why the gradient pass can't be made cheaper by rearranging `@cached`

`GPUArrays.@cached` marks every allocation *busy* for the whole duration of the block and
only recycles buffers back to the *free* pool at the **end** of the block (`free_busy!`).
So **within one gradient pass there is no intra-pass buffer reuse**: two same-shaped buffers
with non-overlapping lifetimes still get two separate cached allocations. The cache's
resident set grows to the *sum of all distinct-shaped allocations in the pass* and stays
pinned. The gradient is a single indivisible Zygote call, so you cannot insert `@cached`
boundaries inside it. CUDA's own stream-ordered pool, by contrast, frees a buffer as soon as
its refcount drops, so it *does* get intra-pass reuse — which is exactly why `:manual` has a
lower peak at large batch.

### The tradeoff is GC-pressure / batch-size dependent

* **Large batch, few steps** (bs128): the cache *inflates* peak (+52%, 26.8 vs 17.6 GiB) and
  can OOM a card the manual loop fits on. The deterministic footprint is bigger than the live
  working set the GC would otherwise keep.
* **Small batch, many steps** (bs32): the cache *halves* peak (8.3 vs 17.1 GiB), because it
  replaces GC-lagged garbage (which piles up to 17 GiB before collection) with a stable
  6.7 GiB footprint.

In short, `@cached` trades GC non-determinism for a fixed resident set equal to one gradient
pass's total distinct allocations — a win for high-GC-pressure loops (small batch / many
steps, or the "few large repeated-shape buffers" pattern it was designed for), a
pessimisation for large whole-model steps.
