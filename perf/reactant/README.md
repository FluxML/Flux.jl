# Zygote vs Reactant training benchmark

Compares three ways of training the same small-image **ResNet-18** — Flux's default **Zygote**
eager path (as a bare loop and via `Flux.train!`) and a **Reactant**-compiled step — reporting
**time per step** and **peak GPU memory** at a couple of batch sizes.

| Backend | Gradient | Execution |
|---|---|---|
| Zygote          | `Flux.withgradient(loss, model, x, y)` in a bare loop | eager CUDA.jl kernels, op-by-op, no GC management |
| Zygote `train!` | same, driven through `Flux.train!` | eager op-by-op, plus `train!`'s adaptive `GC.gc(false)` each step |
| Reactant        | `Flux.withgradient(loss, AutoEnzyme(), model, …)` | whole step traced once and `@compile`d to one XLA executable |

The Reactant step compiles the *entire* training step — forward pass, Enzyme reverse pass, and
the `Optimisers.update!` — into a single XLA program, so XLA can fuse kernels and plan memory
ahead. The Zygote step dispatches every operation at run time.

The **`Flux.train!`** row runs the *same* eager Zygote gradient as the bare-loop row, but through
Flux's training loop, whose default `gc_interval = :auto` fires an incremental `GC.gc(false)`
adaptively (every step, once steps are compute-bound) to reclaim dead GPU buffers. Comparing it
against the bare loop isolates what that adaptive GC buys.

## Running

```bash
# first time only
julia --project=perf/reactant -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'

# -t2 so the eager memory sampler has a thread of its own (see "Memory is measured per-backend")
julia -t2 --project=perf/reactant perf/reactant/benchmark.jl
```

The project uses the in-repo Flux (`[sources] Flux = {path = "../.."}`), so it benchmarks your
working copy. With no functional CUDA GPU it still runs on CPU — Reactant compiles and trains
there too — but only the timings are meaningful and the memory column reads `n/a`.

## Sample output

On an **NVIDIA GeForce RTX 5090** (32 GiB), ResNet-18, 20 timed steps, `Adam(1f-3)`:

```
● ResNet-18, batch 64   (20 timed steps)
  backend             time/step     peak GPU mem     loss (start → end)
  Zygote               19.25 ms       18.283 GiB          5.617 → 0.000
  Zygote train!        19.05 ms        4.283 GiB          5.617 → 0.000
  Reactant             19.55 ms        1.335 GiB          5.617 → 0.000

● ResNet-18, batch 128   (20 timed steps)
  backend             time/step     peak GPU mem     loss (start → end)
  Zygote              216.71 ms        7.000 GiB          5.746 → 0.001
  Zygote train!       213.07 ms        6.625 GiB          5.746 → 0.001
  Reactant             38.32 ms        2.550 GiB          5.746 → 0.001
```

The `loss (start → end)` column is a sanity check that each backend is actually training: all
start at the same `log(200) ≈ 5.6` (identical `deepcopy`'d init, probed with batch statistics) and
`Adam` drives them to ~0 by memorising the single repeated batch. Crucially all three rows now
*agree*: every backend is put in `Flux.trainmode!` so the probe reads BatchNorm's *batch*
statistics. Without that, a bare `withgradient` loop (and the Reactant step) would probe in eval
mode using *running* statistics — which lag badly on a single repeated batch — and report a much
higher loss than `Flux.train!` (which forces `trainmode!` internally), making the same
optimisation look wildly different for a measurement reason, not a training one. Three performance
things stand out:

- **Adaptive GC is a large memory win at no time cost (batch 64).** `Flux.train!` drops peak from
  **18.3 → 4.28 GiB (4.3×)** versus the bare loop while the time is unchanged (19.05 vs 19.25 ms):
  the bare loop's 18.3 GiB is 20 steps of un-reclaimed dead buffers piling up, and `train!`'s
  per-step incremental GC reclaims them and hides fully behind the compute.
- **But adaptive GC does *not* fix the batch-128 cliff** (213.1 vs 216.7 ms). That slowdown is not
  dead-buffer accumulation but memory-pressure *thrashing at allocation time* (CUDA.jl doing
  synchronous, blocking reclaim when the pool is exhausted mid-step), which a between-steps
  `GC.gc(false)` cannot prevent. The ~7 GiB device-level reading there is understated: under
  memory pressure CUDA.jl reclaims and re-allocates mid-step, so the sampled `total − free`
  dips below the true peak.
- **Reactant wins, especially at scale.** At batch 128 it is **~5.6× faster** (38 vs 213 ms) and
  uses **~2.6× less memory** (2.6 vs 6.6 GiB) than either eager path, because it plans the whole
  step's memory ahead and never hits the allocator cliff.
