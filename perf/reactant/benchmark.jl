# Benchmark: Zygote training vs Reactant training for a ResNet-18 on CUDA.
#
# Two ways to train the same small-image ResNet-18 are compared, step for step:
#
#   * Zygote   — Flux's default eager path. `Flux.withgradient(loss, model, x, y)` runs the
#                CUDA.jl kernels for the forward/backward pass, then `Optimisers.update!`
#                applies the step. Nothing is compiled; every op dispatches at run time.
#   * Reactant — the whole training step (forward, Enzyme reverse pass, and the optimiser
#                update) is traced once and compiled to a single XLA executable with
#                `@compile`, then called each step. XLA fuses kernels and plans memory ahead.
#
# For each backend we report **time per step** (after warm-up / compilation) and **peak GPU
# memory** during the timed steps, at a couple of batch sizes.
#
# Two caveats this script bakes in, both learned the hard way:
#
#   1. Optimiser choice. Reactant traces the optimiser update too, and `Adam`'s state carries a
#      `Tuple{Float32,Float32}` of β-powers that is *decayed and written back* every step; under
#      tracing that write hits `Float32(::TracedRNumber)` and fails to compile (as of
#      Reactant 0.2 / Optimisers 0.4). Optimisers whose state is arrays only — `Descent`,
#      `Momentum` — trace cleanly. We use `Momentum` for *both* backends so the comparison is
#      apples-to-apples; the optimiser is a negligible fraction of a ResNet step regardless.
#
#   2. Memory measurement is per-backend, because the two backends use different allocators and
#      no single axis sees both. The eager backends (Zygote, `train!`) allocate through CUDA.jl's
#      pool, which grows the driver reservation on demand, so *device-level* used memory
#      (`CUDA.memory_info`, driver total−free), sampled from a background task, tracks their
#      working set. Reactant allocates through XLA's BFC pool, which grabs a big slab up front and
#      sub-allocates inside it — device-level memory stays pinned at the slab size and reports ~0
#      change — so we read XLA's *own* allocator high-water mark (`Reactant.XLA.allocatorstats`,
#      `peak_bytes_in_use`). Both are baselined to the increase over what was resident before the
#      timed run. (Even with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, XLA still reserves ~75 % of the
#      card as its pool; the allocator-stats path measures the true working set regardless.) See
#      the "GPU memory helpers" section.
#
# Run with:
#
#     julia --project=perf/reactant perf/reactant/benchmark.jl
#
# On the first run, instantiate the environment:
#
#     julia --project=perf/reactant -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'
#
# With no functional CUDA GPU the script still runs on CPU (Reactant compiles and trains on
# CPU too); only the timings are meaningful there, and the memory columns read "n/a".

# --- XLA/Reactant memory knobs: must be set before Reactant initialises its client ---------
# Keep XLA from eagerly pre-grabbing most of the card. `MEM_FRACTION` caps how much XLA may grow
# into. (XLA still reserves a large pool; the per-backend memory measurement reads XLA's own
# allocator counters rather than device-level memory, so this no longer affects the reported peak.)
get!(ENV, "XLA_PYTHON_CLIENT_PREALLOCATE", "false")
get!(ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

using Flux
using Reactant
using Enzyme
using Optimisers
using CUDA, cuDNN
using MLDataDevices
using Statistics: mean
using Printf: @printf, @sprintf

const HAS_GPU = CUDA.functional()

# ---------------------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------------------

# Peak-memory measurement is per-backend, because the two backends allocate through different
# pools and no single axis sees both faithfully:
#
#   * Zygote / `train!` allocate through CUDA.jl's pool, which grows its driver reservation on
#     demand — so device-level used memory (driver `total − free`) tracks their working set, and
#     `with_peak_mem` samples it. (See `zygote_backend` / `train_zygote_backend`.)
#   * Reactant allocates through XLA's BFC pool, which grabs a big slab from the driver *up front*
#     and then sub-allocates inside it without further driver calls. Device-level used memory
#     therefore stays pinned at the reserved-slab size and reports ~0 change no matter how much
#     XLA actually uses — so we read XLA's own allocator high-water mark instead. (See
#     `reactant_backend`.)
#
# Each backend exposes a `peakmem(f)` that runs `f()` and returns `(result, peak_bytes)`, where
# the peak is baselined to the *increase* over what was already resident when `f` started (model
# weights, optimiser state, compiled constants) — i.e. the timed run's transient working set.

# Device-level used bytes from the driver (total − free), for the CUDA.jl-pool (eager) backends.
device_used_bytes() = HAS_GPU ? (let (free, total) = CUDA.memory_info(); total - free end) : 0

fmt_bytes(n) = HAS_GPU ? Base.format_bytes(n) : "n/a"

"""
    with_peak_mem(f)

Run `f()` while a background task polls device-level used memory, and return
`(result, peak_used_bytes)` where the peak is the high-water mark reached during `f` (baselined
so it measures the *increase* over what was already resident when `f` started). Sampling is the
only option for the CUDA.jl pool — a short-lived peak between samples can be missed, so treat the
number as a tight lower bound on the true peak. Used by the eager (Zygote / `train!`) backends.
"""
function with_peak_mem(f)
    HAS_GPU || return (f(), 0)
    GC.gc(); CUDA.reclaim()
    base = device_used_bytes()
    peak = Ref(base)
    stop = Ref(false)
    sampler = Threads.@spawn begin
        while !stop[]
            u = device_used_bytes()
            u > peak[] && (peak[] = u)
            sleep(0.001)
        end
    end
    try
        result = f()
        return (result, max(0, peak[] - base))
    finally
        stop[] = true
        wait(sampler)
    end
end

"""
    reactant_peak_mem(f)

Run `f()` and return `(result, peak_bytes)` using XLA's *own* BFC-allocator counters rather than
device-level memory (which can't see inside XLA's pre-grabbed pool). `peak_bytes_in_use` is a
lifetime high-water mark; we baseline it against `bytes_in_use` sampled just before `f` (the
resident model / optimiser / constants after warm-up) so the result is the timed run's working-
set increase — the same "transient over resident" quantity `with_peak_mem` reports for the eager
backends. Because the mark is lifetime, an unusually large *compile-time* scratch peak (before
`f`) could inflate it; for a ResNet the step's activation set dwarfs any such scratch, so this
tracks the true per-step working set closely.
"""
function reactant_peak_mem(f)
    HAS_GPU || return (f(), 0)
    before = Reactant.XLA.allocatorstats().bytes_in_use
    result = f()
    peak = Reactant.XLA.allocatorstats().peak_bytes_in_use
    return (result, max(0, peak - before))
end

# ---------------------------------------------------------------------------------------
# ResNet-18, small-image variant (mirrors examples/resnet_tinyimagenet and the model used in
# perf/caching_allocator — a deep, allocation-heavy conv net, the interesting case for both
# XLA fusion and memory).
# ---------------------------------------------------------------------------------------

const NCLASSES = 200

struct BasicBlock{C,S}
    convs::C
    shortcut::S
end
Flux.@layer BasicBlock
(m::BasicBlock)(x) = relu.(m.convs(x) .+ m.shortcut(x))

function BasicBlock(inplanes::Int, planes::Int; stride::Int = 1)
    convs = Chain(
        Conv((3, 3), inplanes => planes; stride, pad = 1, bias = false), BatchNorm(planes, relu),
        Conv((3, 3), planes => planes; pad = 1, bias = false), BatchNorm(planes),
    )
    shortcut = if stride != 1 || inplanes != planes
        Chain(Conv((1, 1), inplanes => planes; stride, bias = false), BatchNorm(planes))
    else
        identity
    end
    return BasicBlock(convs, shortcut)
end

function resnet_stage(inplanes, planes, nblocks; stride)
    blocks = Any[BasicBlock(inplanes, planes; stride)]
    for _ in 2:nblocks
        push!(blocks, BasicBlock(planes, planes))
    end
    return Chain(blocks...)
end

function resnet18(; nclasses = NCLASSES)
    return Chain(
        Conv((3, 3), 3 => 64; pad = 1, bias = false), BatchNorm(64, relu),
        resnet_stage(64, 64, 2; stride = 1),
        resnet_stage(64, 128, 2; stride = 2),
        resnet_stage(128, 256, 2; stride = 2),
        resnet_stage(256, 512, 2; stride = 2),
        AdaptiveMeanPool((1, 1)), Flux.flatten, Dense(512 => nclasses),
    )
end

# One synthetic Tiny-ImageNet-shaped batch (64x64 RGB, 200 classes). A single fixed batch is
# reused every step: we are timing the compute, not a data pipeline.
make_batch(bs) = (randn(Float32, 64, 64, 3, bs), Flux.onehotbatch(rand(1:NCLASSES, bs), 1:NCLASSES) .* 1.0f0)

loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

# ---------------------------------------------------------------------------------------
# Backends. Each returns a `(step!, sync, lossof, handles...)` bundle so the driver can treat
# them uniformly: `step!()` does one in-place training step, `sync()` blocks until the device
# has finished all queued work (so timing is honest), and `lossof()` reads the current loss.
# ---------------------------------------------------------------------------------------

# Grab the first Conv weight as a cheap thing to block on / read back for synchronisation.
first_weight(m) = m.layers[1].weight

function zygote_backend(model0, batch; lr)
    dev = HAS_GPU ? gpu_device() : cpu_device()
    model = model0 |> dev
    x, y = batch[1] |> dev, batch[2] |> dev
    opt = Flux.setup(Descent(lr), model)
    step!() = begin
        _, gs = Flux.withgradient(m -> loss(m, x, y), model)
        Optimisers.update!(opt, model, gs[1])
        return nothing
    end
    run!(n) = (for _ in 1:n; step!(); end; nothing)
    sync() = HAS_GPU ? CUDA.synchronize() : nothing
    lossof() = loss(model, x, y)
    return (; run!, sync, lossof, model, peakmem = with_peak_mem)
end

# Same eager Zygote gradient as `zygote_backend`, but driven through `Flux.train!` instead of a
# bare loop. `train!`'s default `gc_interval = :auto` fires an incremental `GC.gc(false)`
# adaptively — for compute-bound steps (longer than a few ms) that means *every* step — to
# reclaim dead GPU buffers so reserved memory doesn't creep up under the eager allocator
# (issue #2523). The manual loop above does no collection, so pairing the two isolates exactly
# what that adaptive GC buys. The adaptive cadence is stateful across steps, so `run!(n)` must
# hand `train!` the whole run of `n` steps in one call (not one step at a time).
function train_zygote_backend(model0, batch; lr)
    dev = HAS_GPU ? gpu_device() : cpu_device()
    model = model0 |> dev
    x, y = batch[1] |> dev, batch[2] |> dev
    opt = Flux.setup(Descent(lr), model)
    # `train!` splats each data item into `loss`, so yield the fixed batch as the tuple `(x, y)`.
    run!(n) = Flux.train!(loss, model, Iterators.repeated((x, y), n), opt)
    sync() = HAS_GPU ? CUDA.synchronize() : nothing
    lossof() = loss(model, x, y)
    return (; run!, sync, lossof, model, peakmem = with_peak_mem)
end

function reactant_backend(model0, batch; lr)
    dev = reactant_device(force = HAS_GPU)
    model = model0 |> dev
    x, y = batch[1] |> dev, batch[2] |> dev
    opt = Flux.setup(Descent(lr), model)
    # The full step — forward, Enzyme reverse pass, optimiser update — compiled to one executable.
    raw_step!(opt, model, x, y) = begin
        _, gs = Flux.withgradient(m -> loss(m, x, y), AutoEnzyme(), model)
        Optimisers.update!(opt, model, gs[1])
        return nothing
    end
    compiled = @compile raw_step!(opt, model, x, y)
    step!() = compiled(opt, model, x, y)
    run!(n) = (for _ in 1:n; compiled(opt, model, x, y); end; nothing)
    # PJRT runs async; reading a device array back to the host blocks until the queue drains.
    sync() = (Array(first_weight(model)); nothing)
    lossof() = Reactant.to_number(Reactant.@jit loss(model, x, y))
    return (; run!, sync, lossof, model, peakmem = reactant_peak_mem)
end

# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------

"""
    time_backend(make_backend, model0, batch; steps, warmup) -> (; time_per_step, peak_used, loss0, loss1)

Build a backend, run `warmup` untimed steps (for Reactant the first call is the compile, which
must not be timed), then time `steps` in-place training steps — synchronising once at the end
so async device work is included — while tracking peak device memory. `loss0`/`loss1` are the
loss before and after the timed steps, a sanity check that the model is actually training.
"""
function time_backend(make_backend, model0, batch; steps::Int, warmup::Int)
    b = make_backend(model0, batch)
    b.run!(warmup)
    b.sync()
    loss0 = Float64(b.lossof())

    (t, peak) = b.peakmem() do
        dt = @elapsed begin
            b.run!(steps)
            b.sync()
        end
        return dt
    end

    loss1 = Float64(b.lossof())
    return (; time_per_step = t / steps, peak_used = peak, loss0, loss1)
end

function compare(bs; steps::Int, warmup::Int, lr = 1.0f-2)
    println("\n", "─"^72)
    println("● ResNet-18, batch $bs   ($steps timed steps)")
    println("─"^72)
    @printf("  %-14s %14s %16s %22s\n", "backend", "time/step", "peak GPU mem", "loss (start → end)")

    model0 = resnet18()
    batch = make_batch(bs)
    backends = [
        ("Zygote",        (m, b) -> zygote_backend(m, b; lr)),
        ("Zygote train!", (m, b) -> train_zygote_backend(m, b; lr)),
        ("Reactant",      (m, b) -> reactant_backend(m, b; lr)),
    ]
    for (name, mk) in backends
        r = try
            time_backend(mk, deepcopy(model0), batch; steps, warmup)
        catch e
            (e isa OutOfGPUMemoryError) || rethrow()
            nothing
        end
        if r === nothing
            @printf("  %-14s %14s %16s %22s\n", name, "—", "OOM", "—")
        else
            @printf("  %-14s %14s %16s %22s\n", name,
                    @sprintf("%.2f ms", 1e3 * r.time_per_step),
                    fmt_bytes(r.peak_used),
                    @sprintf("%.3f → %.3f", r.loss0, r.loss1))
        end
    end
    return nothing
end

function main()
    if HAS_GPU
        free, total = CUDA.memory_info()
        @info "Running on GPU" CUDA.name(CUDA.device()) free = Base.format_bytes(free) total = Base.format_bytes(total)
        Threads.nthreads() == 1 &&
            @warn "Started with a single thread: the memory sampler shares it with the training \
                   loop and may under-sample the peak. Re-run with `julia -t2` (or more) for \
                   reliable peak-memory numbers."
    else
        @info "No functional CUDA GPU: running on CPU (Reactant still compiles; only timings are meaningful, memory reads n/a)."
    end

    # Small batches by default so this fits any card; bump these on a big GPU.
    batchsizes = HAS_GPU ? (64, 128) : (8,)
    steps  = HAS_GPU ? 20 : 3
    warmup = 3
    for bs in batchsizes
        compare(bs; steps, warmup)
    end
    println("\nDone.")
    return nothing
end

main()
