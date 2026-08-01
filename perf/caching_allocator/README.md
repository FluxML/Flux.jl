# Caching allocator benchmark

`Flux.train!` wraps each training step in a
[`GPUArrays.AllocCache`](https://juliagpu.github.io/GPUArrays.jl/) so the GPU memory
allocated during one step is reused by the next, keeping the memory footprint stable and
avoiding the ever-growing GPU memory usage reported in
[issue #2523](https://github.com/FluxML/Flux.jl/issues/2523). Because the caching
allocator can occasionally slow training down, `train!` exposes a `caching_allocator`
keyword to turn it off:

```julia
Flux.train!(loss, model, data, opt; caching_allocator = false)
```

`benchmark.jl` measures the time / memory trade-off of the two settings across a few
models (the minimal MLP from #2523, a deeper MLP, a small CNN, and a tiny model where the
allocator bookkeeping dominates), and reproduces the forward-pass memory-growth
observation from the issue.

## Running

```bash
# first time only
julia --project=perf/caching_allocator -e 'using Pkg; Pkg.instantiate()'

julia --project=perf/caching_allocator perf/caching_allocator/benchmark.jl
```

The project uses the in-repo Flux (`[sources] Flux = {path = "../.."}`), so it benchmarks
your working copy. It runs on CPU when no functional CUDA GPU is available, in which case
only the timings are meaningful.

## Sample output (RTX 5090)

```
● Deep MLP, batch 256
  caching_allocator          time/epoch      peak reserved
  false (off)                 661.45 ms          1.406 GiB
  true  (on)                  651.99 ms         32.000 MiB
  → caching allocator is 1.01× faster on time, peak reserved memory 0.02× of off
```

The caching allocator keeps peak reserved GPU memory small (here 32 MiB vs 1.4 GiB) at
roughly neutral time. Whether it helps or hurts throughput is workload-dependent, so use
this script to check your own models before deciding to pass `caching_allocator = false`.
