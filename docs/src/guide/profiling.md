# [Profiling a Training Loop](@id man-profiling)

A profiler shows where a training loop spends its time, and whether the GPU waits for the host.
On NVIDIA GPUs, [`train!`](@ref Flux.train!), [`trainstep!`](@ref Flux.trainstep!) and
[`trainstep_withgradient!`](@ref Flux.trainstep_withgradient!) mark the phases of each step with
[NVTX](https://github.com/JuliaGPU/NVTX.jl) ranges in a `Flux` domain. CUDA.jl's integrated
profiler and NVIDIA Nsight Systems show them alongside the CUDA API calls and the kernels:

| Range | What it covers |
|:------|:---------------|
| `trainstep!`, `trainstep_withgradient!` | One call, or one step of `train!`, which shows as `trainstep!`: everything below except `data`. |
| `forward`, `backward` | With Zygote, the forward pass, which computes the loss, and the backward pass, which computes the gradient. |
| `gradient` | With other AD backends, such as Enzyme or Mooncake, both passes. |
| `update` | The optimiser update. |
| `incremental GC`, `full GC` | The garbage collection after the step, as set by `gc_interval` (see [`train!`](@ref Flux.train!)). |
| `data` | In `train!`, fetching the next batch from `data`. For a `DataLoader` wrapped with a device, this includes moving the batch to the GPU. |

The ranges are defined in a package extension on NVTX.jl, which CUDA.jl loads. With no profiler
attached, they add about 0.1 μs to a step.

The ranges mark time on the host. GPU kernels run asynchronously, so a range ends once the host has
queued its work, unless the host waits for the result. With CUDA.jl, the forward pass usually
waits, because the loss is read back to the host. So the GPU time of a step's backward pass and update
can show up in the next step's `forward` range.

## Profiling a few steps with CUDA.jl

First warm up. The first step with given types compiles, which can take minutes, and on a GPU it also
runs cuDNN's search for convolution algorithms. Then profile a few steps with `CUDA.@profile`.
Profile exactly the call you warmed up, with arguments of the same types, or it compiles again inside
the profile:

```julia
using Flux, CUDA, cuDNN

dev = gpu_device()
model = Chain(Dense(784 => 4096, relu), Dense(4096 => 4096, relu), Dense(4096 => 10)) |> dev
opt_state = Flux.setup(Adam(), model)
x, y = randn(Float32, 784, 16384), Flux.onehotbatch(rand(1:10, 16384), 1:10)
loader = dev(Flux.DataLoader((x, y); batchsize = 2048, shuffle = true))
loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

batches = Iterators.take(loader, 3)
for _ in 1:3  # warm up
    Flux.train!(loss, model, batches, opt_state)
end

CUDA.@profile begin
    GC.enable_finalizers()  # see below
    try
        Flux.train!(loss, model, batches, opt_state)
    finally
        GC.disable_finalizers()
    end
end
```

`CUDA.@profile` runs the code with finalizers disabled ([CUDA.jl#3530](https://github.com/JuliaGPU/CUDA.jl/issues/3530)),
so no GPU array is freed while it runs, however often the GC runs. Profiling many steps, or a large
model, can then run out of GPU memory. The calls to `GC.enable_finalizers` and `GC.disable_finalizers`
turn finalizers back on inside the profiled code.

The result starts with a summary of the CUDA API calls made by the host, then of the operations run
by the GPU, then of the NVTX ranges:

```
Profiler ran for 39.47 ms, capturing 4033 events.

Host-side activity: calling CUDA APIs took 3.5 ms (8.87% of the trace)
[...]

Device-side activity: GPU was busy for 13.19 ms (33.43% of the trace)
[...]

NVTX ranges:
 Time (%) │ Total time │ Calls │ Time distribution                    │ Name
──────────┼────────────┼───────┼──────────────────────────────────────┼─────────────────────
   63.08% │    24.9 ms │     3 │    8.3 ms ± 10.32  (  2.27 ‥ 20.21)  │ Flux.trainstep!
   42.79% │   16.89 ms │     1 │                                      │ Flux.incremental GC
   18.08% │    7.14 ms │     4 │   1.78 ms ± 1.29   (   0.0 ‥ 2.74)   │ Flux.data
   15.47% │    6.11 ms │     3 │   2.04 ms ± 0.48   (  1.75 ‥ 2.59)   │ Flux.forward
    3.37% │    1.33 ms │     3 │ 443.22 µs ± 102.14 (326.48 ‥ 516.13) │ Flux.backward
    1.32% │  519.27 µs │     3 │ 173.09 µs ± 41.59  (133.92 ‥ 216.74) │ Flux.update
```

With `CUDA.@profile trace = true`, each event is listed in order instead, which helps to find one slow
step. The NVTX ranges then show when each one started, and for how long it ran:

```
NVTX ranges:
 ID │    Start │      Time │ Domain │ Name       │ Payload
────┼──────────┼───────────┼────────┼────────────┼─────────
 37 │  5.82 ms │   2.47 ms │   Flux │ data       │       0
 38 │   8.3 ms │   3.88 ms │   Flux │ trainstep! │       0
 39 │  8.31 ms │   2.89 ms │   Flux │ forward    │       0
 40 │  11.2 ms │ 746.58 µs │   Flux │ backward   │       0
 41 │ 11.95 ms │  219.4 µs │   Flux │ update     │       0
 42 │ 12.18 ms │   2.54 ms │   Flux │ data       │       0
 43 │ 14.73 ms │   2.26 ms │   Flux │ trainstep! │       0
[...]
```

The `ID` column of the host and device tables links each API call to the operation it launched.
`show(stdout, "application/json", result)` prints the result as JSON, for processing by a script.

## Reading the result

Start from the line `GPU was busy for ... (...% of the trace)`. For the rest of the trace, the GPU
was idle. If it was busy for most of the trace, the loop is limited by the GPU: faster host code
won't help, but a lower precision (see [`bf16`](@ref Flux.bf16)) or a better kernel would.
The host-side table then shows much of the time in `cuEventSynchronize` or `cuStreamSynchronize`,
where the host waits for the GPU.

If the GPU was idle for much of the trace, the NVTX ranges show where the host spent that time:

* A long `data` range means that fetching batches keeps the GPU waiting.
  `DataLoader(...; parallel = true)` prepares them on several threads, if Julia was started with
  several (see also [Transferring Training Data](@ref)).
* Long `incremental GC` or `full GC` ranges mean that the collections after each step don't overlap
  with GPU work. A larger `gc_interval` makes them less frequent, at the cost of more GPU memory in
  use (see [`train!`](@ref Flux.train!)).
* A `forward` or `backward` range much longer than the GPU time of its kernels means that the host
  can't launch them fast enough, for example because the model has many small operations, or reads
  values back to the host in the middle of a pass (`cuMemcpyDtoHAsync` in the host-side table).
  Larger batches, or compiling the step with [Reactant](reactant.md), help.

In the example above, the GPU was busy for a third of the trace. Fetching each batch, which shuffles
the data on the CPU and copies the batch to the GPU, takes about as long as the step itself, and one
collection took 17 ms.

## Julia's garbage collections

Julia's GC can also run in the middle of a step, when Julia decides to collect, or when CUDA.jl
collects because GPU memory is getting full. To see these collections, set the environment
variable `JULIA_NVTX_CALLBACKS=gc` before starting Julia (or at least before the first
`CUDA.@profile` of the session). Every collection then shows as a `Julia.GC` range. A `Julia.GC`
range outside of a `Flux.incremental GC` or `Flux.full GC` range is one that Flux didn't run.

## Nsight Systems

For a timeline, use [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems), as described
in [CUDA.jl's documentation](https://cuda.juliagpu.org/stable/development/profiling/). Start Julia
with `nsys launch julia`, warm up, and wrap the steps to profile in `CUDA.@profile external = true`.
The `Flux` ranges appear in the timeline, along with the CUDA API calls and kernels.
The finalizer workaround above isn't needed with an external profiler.

## Other backends

With Enzyme or another AD backend than Zygote, the forward and backward passes show as one
`gradient` range.

On a Reactant device, the whole step is compiled into a single executable, and only the `data` and
`trainstep!` ranges appear. `trainstep!` covers queuing the step, which returns before the device
has run it. To profile what runs on the device, use Reactant's profiler, `Reactant.with_profiler`.
NVTX ranges are only recorded when NVTX.jl is loaded: without CUDA.jl, load it with `using NVTX`
to see Flux's ranges in Nsight Systems.
