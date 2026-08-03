# Autocast benchmark

Benchmarks Flux's [`autocast`](../../src/autocast.jl) mixed-precision training on the GPU,
using ResNet-18 (the compute- and allocation-heavy conv net where mixed precision matters
most).

`autocast(model, T)` — with `T = Float16` or `BFloat16` — wraps the model so that its
matmul/convolution-heavy layers cast down to `T` at call time while the parameters stay in
`Float32` (PyTorch `torch.autocast` semantics). The same wrapping happens automatically
when you pass `autocast=T` to `Flux.gradient` / `Flux.train!`.

The script reports two things:

1. **Speedup and memory saving.** Forward pass, forward+backward step, and peak GPU memory
   for the `Float32` baseline vs `autocast=Float16` / `autocast=BFloat16`. Half precision
   should run the kernels faster and roughly halve the peak activation memory.

2. **Wrapper overhead.** The wrapper is a differentiable model transform, so the
   `autocast=T` keyword rebuilds *and back-propagates through* the wrapper tree on every
   `gradient` call. This compares:
   - wrapping the model **once** (`wm = autocast(model, T)`, reused every step), against
   - wrapping it **each time** inside the differentiated closure (the keyword path),

   plus the isolated cost of the `autocast(model, T)` construction itself. The delta tells
   you whether pre-wrapping once (and reusing the wrapped model) is worth it, or whether
   the convenience keyword's per-step cost is negligible against the ResNet step.

## Running

```
# first run: instantiate the environment
julia --project=perf/autocast -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'

julia --project=perf/autocast perf/autocast/benchmark.jl
```

Requires a functional NVIDIA GPU to be meaningful; without one it still runs (CPU timings
only, memory columns blank).

## PyTorch reference

[`benchmark_pytorch.py`](benchmark_pytorch.py) runs the identical model, batch sizes, and
timing/peak-memory methodology under PyTorch's `torch.autocast` (AMP), so Flux's numbers can
be compared against PyTorch's native mixed precision on the same GPU:

```
uv run --python 3.12 --index-url https://download.pytorch.org/whl/cu128 \
    --with torch perf/autocast/benchmark_pytorch.py
```

(Blackwell GPUs need the `cu128` wheels, i.e. `torch >= 2.7`.)
