# Caching allocator benchmark

`Flux.train!` manages GPU memory to avoid the ever-growing usage reported in
[issue #2523](https://github.com/FluxML/Flux.jl/issues/2523). By default
(`caching_allocator = false, gc_interval = :auto`) it runs without a cross-step buffer cache and
instead paces an incremental `GC.gc(false)` adaptively, keeping memory at the working set.
Alternatively, `caching_allocator = true` reuses buffers with a
[`GPUArrays.AllocCache`](https://juliagpu.github.io/GPUArrays.jl/): that keeps *reserved* memory
flat and is cheap for small models, but within a step it *pins* every allocation until the step
ends, so a step's peak becomes the **sum** of its allocations rather than its working set — which
inflates memory for allocation-heavy models (deep conv nets) and can OOM.

`train!` exposes these keywords to steer it:

```julia
Flux.train!(loss, model, data, opt; caching_allocator = false)  # plain pool + GC
Flux.train!(loss, model, data, opt; caching_allocator = false, gc_interval = 1)      # paced GC every step
Flux.train!(loss, model, data, opt; caching_allocator = false, gc_interval = :auto)  # adaptive paced GC
```

`gc_interval = N` issues an incremental `GC.gc(false)` every `N` steps. With the cache off,
dead GPU buffers are otherwise only reclaimed by the GC, which rarely fires on its own (the
`CuArray` wrappers are tiny on the CPU heap), so reserved memory creeps up (#2523); a paced
GC bounds that growth at a fraction of the cost of collecting every step. `gc_interval = :auto`
picks the cadence from wall-clock timing alone (no GPU/backend queries, so it works on any
backend): a compute-bound step hides an incremental GC, so it collects every step and keeps
memory minimal; a cheap step collects rarely, holding the amortized GC cost near ~2%.

`benchmark.jl` compares five policies — `cache on`, `cache off`, `cache off + gc/1`,
`cache off + gc/4`, `cache off + gc/auto` — across a range of models (the minimal MLP from
#2523, a deeper MLP, a small CNN, a tiny model where allocator bookkeeping dominates, and a
small-image ResNet-18 at batch 64/128). It also reproduces the #2523 forward-pass memory
growth and runs a ResNet-18 cold-start check (cold vs warm) confirming that `train!`'s
first-step cache skip keeps cuDNN's one-off algorithm-search workspaces from being pinned.

## Running

```bash
# first time only
julia --project=perf/caching_allocator -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'

julia --project=perf/caching_allocator perf/caching_allocator/benchmark.jl
```

The project uses the in-repo Flux (`[sources] Flux = {path = "../.."}`), so it benchmarks
your working copy. It runs on CPU when no functional CUDA GPU is available, in which case
only the timings are meaningful. Peak `used`/`reserved` are read from the CUDA memory pool's
exact high-water marks (`MEMPOOL_ATTR_USED_MEM_HIGH` / `RESERVED_MEM_HIGH`), reset before each
measured run — no sampling, so the mid-step peak is captured exactly.

## Sample output (RTX 5090)

Small-footprint model — the cache wins (flat memory at no time cost); paced GC bounds memory
too but costs time on a cheap step:

```
● Deep MLP, batch 256
  config                  time/epoch       peak used   peak reserved
  cache on                  15.89 ms     431.498 MiB     448.000 MiB
  cache off                 16.31 ms      16.684 GiB      16.719 GiB
  cache off + gc/1          52.76 ms      91.040 MiB      96.000 MiB
  cache off + gc/4          23.72 ms     176.093 MiB     192.000 MiB
  cache off + gc/auto       16.03 ms     924.560 MiB     928.000 MiB
```

Deep conv net — time is identical across policies (compute-bound, so the GC is hidden), and
`cache off + gc/1` uses far less memory than the cache:

```
● ResNet-18, batch 128
  config                  time/epoch       peak used   peak reserved
  cache on                 193.90 ms      23.187 GiB      23.438 GiB
  cache off                194.26 ms      20.113 GiB      20.250 GiB
  cache off + gc/1         194.07 ms       8.412 GiB       8.938 GiB
  cache off + gc/4         193.66 ms      20.113 GiB      20.281 GiB
  cache off + gc/auto      194.26 ms       8.412 GiB       8.938 GiB
```

`gc/auto` here lands exactly on `gc/1` (the step is compute-bound, so it collects every step)
without being told to — the same setting leaves the sub-millisecond MLP/tiny steps collecting
rarely, so it never pays `gc/1`'s slowdown on them.

Two columns because they say different things: **used** = live bytes (what OOMs), **reserved**
= pool high-water (issue #2523). With the cache on, `used ≈ reserved` — the step's buffers are
genuinely *pinned*. With the cache off and no paced GC, `used ≈ reserved` too, but for a
different reason: the GC never frees the dead buffers, so they stay live (until CUDA's
pressure-GC caps it near the card size). Only paced GC opens a gap (`used < reserved`): dead
buffers are freed back to the pool, which keeps them reserved for cheap reuse.

**Takeaway.** The right policy is workload-dependent:

- **Small / cheap steps (MLPs, #2523):** keep the cache on. It holds memory flat at no time
  cost; paced GC bounds memory too but its cost is a large fraction of a cheap step.
- **Allocation-heavy, compute-bound models (conv nets):** use `caching_allocator = false,
  gc_interval = 1`. The step is compute-bound, so the per-step GC is fully hidden (same
  time/epoch), while the cache's per-step pinning inflates peak used ~2.8× (8.4 → 23.2 GiB here).
  A fixed `gc_interval = 4` is not enough for this model — one big step already fills memory —
  so pace the GC to the per-step allocation size.
- **Don't want to choose?** `caching_allocator = false, gc_interval = :auto` derives the cadence
  from step timing: it matches `gc/1`'s memory on the conv net and cache-off's speed on the
  MLPs, with no per-model tuning and no backend-specific code. It errs toward speed on medium
  steps (the Deep MLP sits at ~0.9 GiB rather than `gc/4`'s 0.18 GiB), so pick a fixed
  `gc_interval` when you want the tightest possible memory on such a model.
