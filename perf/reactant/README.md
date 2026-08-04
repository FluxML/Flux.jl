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

## Two things this benchmark had to work around

### 1. Adam does not compile under Reactant — we use `Momentum`

Reactant traces the optimiser update along with everything else. Stock `Optimisers.Adam` keeps
a `Tuple{Float32,Float32}` of β-powers in its per-leaf state and *decays and writes it back*
every step (`βt .* β`). Under tracing that written-back value is a `TracedRNumber`, and storing
it into the leaf's concrete `Tuple{Float32,Float32}` field hits `Float32(::TracedRNumber)`,
which has no method — compilation fails. Optimisers whose state is arrays only (`Descent`,
`Momentum`) trace cleanly, so this benchmark uses plain **`Descent` (SGD) for both backends** (an
apples-to-apples comparison; the optimiser is a negligible slice of a ResNet step anyway).

**This is a real gap in Flux's Reactant story, not just a benchmark quirk.** Neither Flux nor
`OptimisersReactantExt` (which only patches `_assert_positive_eta` and `AccumGrad`) ships a
Reactant-compatible `Adam`. **Lux** works around it in
`Lux.ReactantCompatibleOptimisers.make_reactant_compatible`, which — *before* `Optimisers.setup`
— swaps `Adam`/`AdamW`/`Momentum`/`Descent` for structurally identical `ReactantAdam` &c. whose
hyperparameters *and* β-accumulator are **tracked Reactant numbers** (`to_rarray(…;
track_numbers=true)`) rather than plain `Float32`. That makes the in-place state write type-
compatible under tracing, and as a bonus lets you adjust the learning rate without recompiling.
Upstreaming this into Optimisers is tracked in
[FluxML/Optimisers.jl#205](https://github.com/FluxML/Optimisers.jl/issues/205); until then, Flux
+ Reactant users are limited to array-state optimisers or must vendor Lux's rules.

### 2. Memory is measured per-backend

The two backends allocate through different pools, and no single axis sees both faithfully, so
the peak-memory number is measured differently for each:

- **Eager backends (Zygote, `train!`)** allocate through CUDA.jl's pool, which grows its driver
  reservation on demand. So their peak is sampled from a background task reading **device-level**
  used memory (`CUDA.memory_info()`, the driver's `total − free`).
- **Reactant** allocates through XLA's BFC pool, which grabs a big slab from the driver *up front*
  and then sub-allocates inside it without further driver calls. Device-level used memory stays
  pinned at the slab size and reports ~0 change no matter how much XLA actually uses — so for
  Reactant we read XLA's *own* allocator high-water mark instead
  (`Reactant.XLA.allocatorstats().peak_bytes_in_use`).

Both are baselined to the *increase* over what was already resident (model weights, optimiser
state, compiled constants) when the timed run started, so they report the run's transient working
set. Even with `XLA_PYTHON_CLIENT_PREALLOCATE=false` XLA still reserves ~75 % of the card as its
pool; the allocator-stats path measures Reactant's true working set regardless.

Two consequences baked into the script:

- **The eager sampler needs a spare thread.** With `julia -t1` it shares the one thread with the
  training loop and can under-sample the peak; run with `julia -t2` (the script warns if not).
  Because it samples, treat the eager peak as a tight lower bound. (Reactant's number comes from a
  counter, not sampling, so it is exact.)
- **Under memory pressure the eager device-level number is unreliable.** When the working set
  approaches the card and CUDA.jl thrashes (reclaiming and re-allocating mid-step — see batch 128
  below), `total − free` is depressed and *understates* the true peak.

## Sample output

On an **NVIDIA GeForce RTX 5090** (32 GiB), ResNet-18, 20 timed steps:

```
● ResNet-18, batch 64   (20 timed steps)
  backend             time/step     peak GPU mem     loss (start → end)
  Zygote               18.78 ms       18.283 GiB          5.292 → 4.741
  Zygote train!        18.57 ms        4.252 GiB          4.502 → 1.099
  Reactant             16.80 ms        1.313 GiB          5.247 → 4.179

● ResNet-18, batch 128   (20 timed steps)
  backend             time/step     peak GPU mem     loss (start → end)
  Zygote              316.40 ms        7.125 GiB          5.307 → 5.032
  Zygote train!       316.61 ms        6.781 GiB          5.136 → 2.711
  Reactant             33.16 ms        2.607 GiB          5.291 → 5.100
```

The `loss (start → end)` column is a sanity check that each backend is actually training (all
start near `log(200) ≈ 5.3` and decrease). Three things stand out:

- **Adaptive GC is a large memory win at no time cost (batch 64).** `Flux.train!` drops peak from
  **18.3 → 4.25 GiB (4.3×)** versus the bare loop while the time is unchanged (18.57 vs 18.78 ms):
  the bare loop's 18.3 GiB is 20 steps of un-reclaimed dead buffers piling up, and `train!`'s
  per-step incremental GC reclaims them and hides fully behind the compute.
- **But adaptive GC does *not* fix the batch-128 cliff** (316.6 vs 316.4 ms). That slowdown is not
  dead-buffer accumulation but memory-pressure *thrashing at allocation time* (CUDA.jl doing
  synchronous, blocking reclaim when the pool is exhausted mid-step), which a between-steps
  `GC.gc(false)` cannot prevent. The ~7 GiB device-level reading there is understated (see the
  memory caveat above).
- **Reactant wins, especially at scale.** At batch 128 it is **~9.5× faster** (33 vs 316 ms) and
  uses **~2.6× less memory** (2.6 vs 6.8 GiB) than either eager path, because it plans the whole
  step's memory ahead and never hits the allocator cliff.

For reference, a CPU smoke test (no GPU — timings only, batch 8) also shows both paths compile and
train and that the compiled step is already several times faster than eager op-by-op:

```
● ResNet-18, batch 8   (10 timed steps)
  backend             time/step     peak GPU mem     loss (start → end)
  Zygote             3374.53 ms              n/a          5.212 → 4.706
  Reactant            486.90 ms              n/a          4.871 → 2.483
```
