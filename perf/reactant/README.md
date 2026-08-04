# Zygote vs Reactant training benchmark

Compares two ways of training the same small-image **ResNet-18** — Flux's default **Zygote**
eager path and a **Reactant**-compiled step — reporting **time per step** and **peak GPU
memory** at a couple of batch sizes.

| Backend | Gradient | Execution |
|---|---|---|
| Zygote   | `Flux.withgradient(loss, model, x, y)` | eager CUDA.jl kernels, op-by-op |
| Reactant | `Flux.withgradient(loss, AutoEnzyme(), model, …)` | whole step traced once and `@compile`d to one XLA executable |

The Reactant step compiles the *entire* training step — forward pass, Enzyme reverse pass, and
the `Optimisers.update!` — into a single XLA program, so XLA can fuse kernels and plan memory
ahead. The Zygote step dispatches every operation at run time.

## Running

```bash
# first time only
julia --project=perf/reactant -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'

# -t2 so the memory sampler has a thread of its own (see "Measuring memory")
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

### 2. Memory is sampled at the device level

Reactant allocates through XLA's own pool, which CUDA.jl's pool high-water marks never see. So
the peak-memory number is sampled from a background task reading **device-level** used memory
(`CUDA.memory_info()`, i.e. the driver's `total − free`), which captures *both* the CUDA.jl pool
(Zygote) and XLA's pool (Reactant) — the only allocator-agnostic axis.

Two consequences baked into the script:

- **XLA preallocation is disabled** (`ENV["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"`, set at the
  very top before Reactant initialises). Otherwise XLA grabs ~75 % of the card up front and the
  device-level number would just report the reservation, not the working set.
- **Sampling needs a spare thread.** With `julia -t1` the sampler shares the one thread with the
  training loop and can under-sample the peak; run with `julia -t2` (the script warns if not).
  Because it samples, treat the reported peak as a tight lower bound.

## Sample output

CPU smoke test (no GPU — timings only, ResNet-18 at batch 8), showing both paths compile and
train and that the compiled step is already several times faster than eager op-by-op:

```
● ResNet-18, batch 8   (10 timed steps)
  backend         time/step     peak GPU mem     loss (start → end)
  Zygote         3374.53 ms              n/a          5.212 → 4.706
  Reactant        486.90 ms              n/a          4.871 → 2.483
```

On CUDA, fill in the `time/step` and `peak GPU mem` columns for batch 64 / 128 from your card;
the `loss (start → end)` column is a sanity check that each backend is actually training (both
should start near `log(200) ≈ 5.3` and decrease).
