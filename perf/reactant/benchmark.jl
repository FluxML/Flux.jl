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
#   2. Memory measurement. Reactant allocates through XLA's own pool, which CUDA.jl's pool
#      high-water marks do not see, so we sample *device-level* used memory (`CUDA.memory_info`,
#      i.e. the driver's total−free) from a background task and keep the peak — a number that
#      captures both allocators. For it to reflect Reactant's true working set rather than a
#      fixed 75 %-of-card reservation, XLA preallocation must be OFF; the `ENV` below does that
#      and MUST be set before Reactant/XLA initialise, so keep it at the very top of the file.
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
# Turn off XLA's eager pre-grab of most of the card so device-level memory tracks the working
# set. `MEM_FRACTION` caps how much XLA may grow into if it does allocate.
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

# Device-level used bytes from the driver (total − free). Unlike CUDA.jl's pool counters this
# sees *every* context's allocations on the device, so it captures both the CUDA.jl pool (used
# by Zygote) and XLA's pool (used by Reactant) — the only apples-to-apples memory axis here.
device_used_bytes() = HAS_GPU ? (let (free, total) = CUDA.memory_info(); total - free end) : 0

fmt_bytes(n) = HAS_GPU ? Base.format_bytes(n) : "n/a"

"""
    with_peak_mem(f)

Run `f()` while a background task polls device-level used memory, and return
`(result, peak_used_bytes)` where the peak is the high-water mark reached during `f` (baselined
so it measures the *increase* over what was already resident when `f` started). Sampling is the
only allocator-agnostic option — a short-lived peak between samples can be missed, so treat the
number as a tight lower bound on the true peak.
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
    sync() = HAS_GPU ? CUDA.synchronize() : nothing
    lossof() = loss(model, x, y)
    return (; step!, sync, lossof, model)
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
    # PJRT runs async; reading a device array back to the host blocks until the queue drains.
    sync() = (Array(first_weight(model)); nothing)
    lossof() = Reactant.to_number(Reactant.@jit loss(model, x, y))
    return (; step!, sync, lossof, model)
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
    for _ in 1:warmup
        b.step!()
    end
    b.sync()
    loss0 = Float64(b.lossof())

    (t, peak) = with_peak_mem() do
        dt = @elapsed begin
            for _ in 1:steps
                b.step!()
            end
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
    @printf("  %-10s %14s %16s %22s\n", "backend", "time/step", "peak GPU mem", "loss (start → end)")

    model0 = resnet18()
    batch = make_batch(bs)
    backends = [
        ("Zygote",   (m, b) -> zygote_backend(m, b; lr)),
        ("Reactant", (m, b) -> reactant_backend(m, b; lr)),
    ]
    for (name, mk) in backends
        r = try
            time_backend(mk, deepcopy(model0), batch; steps, warmup)
        catch e
            (e isa OutOfGPUMemoryError) || rethrow()
            nothing
        end
        if r === nothing
            @printf("  %-10s %14s %16s %22s\n", name, "—", "OOM", "—")
        else
            @printf("  %-10s %14s %16s %22s\n", name,
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
