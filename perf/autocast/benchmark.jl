# Benchmark for Flux's `autocast` mixed-precision training on the GPU.
#
# `autocast(model, T)` (with T = `Float16` or `BFloat16`) returns a *wrapped* model that
# keeps its parameters in `Float32` but casts the matmul/convolution-heavy layers down to
# `T` at call time, so the compute-intensive kernels run at half precision while the
# parameters stay full precision (PyTorch's `torch.autocast` semantics). The same wrapping
# happens automatically when you pass `autocast=T` to `Flux.gradient` / `Flux.train!`.
#
# This script measures two things on ResNet-18 (the allocation- and compute-heavy conv net
# where mixed precision matters most):
#
#   1. The SPEEDUP and MEMORY SAVING from autocast, versus the plain `Float32` baseline —
#      forward pass, forward+backward step, and peak GPU memory.
#
#   2. The OVERHEAD of the wrapper itself, comparing:
#        * wrapping the model ONCE (`wm = autocast(model, T)`, reused every step), against
#        * wrapping it EACH TIME inside the differentiated closure (what the `autocast=T`
#          keyword does — it rebuilds and differentiates through the wrapper tree on every
#          `gradient` call).
#      The delta isolates the cost of constructing (and back-propagating through) the
#      wrapper tree per step, so we know whether the convenience keyword is worth its price.
#
# Timings are wall-clock medians with a GPU sync; peak memory is read from the CUDA pool's
# exact high-water mark. Requires an NVIDIA GPU to be meaningful (falls back to CPU-only
# timings, with memory columns blank, otherwise).
#
# Run with:
#
#     julia --project=perf/autocast perf/autocast/benchmark.jl
#
# On the first run, instantiate the environment:
#
#     julia --project=perf/autocast -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'

using Flux
using CUDA, cuDNN
using Statistics: median
using Printf: @printf, @sprintf

const HAS_GPU = CUDA.functional()
const DEVICE = HAS_GPU ? gpu : identity

# ---------------------------------------------------------------------------------------
# Timing + GPU-memory helpers
# ---------------------------------------------------------------------------------------

# Median wall-clock time of `f()` in seconds, with a GPU sync folded in so we time the
# whole (possibly asynchronous) GPU work, not just the kernel launches. A few warmups first
# so compilation and pool growth don't pollute the measurement.
function timed(f; samples::Int = 30, warmup::Int = 5)
    for _ in 1:warmup
        HAS_GPU ? CUDA.@sync(f()) : f()
    end
    ts = Float64[]
    for _ in 1:samples
        GC.gc(false)
        push!(ts, HAS_GPU ? (@elapsed CUDA.@sync f()) : (@elapsed f()))
    end
    return median(ts)
end

# Exact peak of *live* (used) pool bytes reached while running `f()` — the number that
# determines an OOM. Reset the pool high-water mark, run, read it back.
gpu_pool() = CUDA.CUDACore.pool_create(CUDA.device())

function peak_used(f; warmup::Bool = true)
    if !HAS_GPU
        f()
        return 0
    end
    warmup && (CUDA.@sync f(); GC.gc(true); CUDA.reclaim())
    CUDA.synchronize()
    CUDA.attribute!(gpu_pool(), CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
    CUDA.@sync f()
    CUDA.synchronize()
    p = Int(CUDA.attribute(UInt64, gpu_pool(), CUDA.MEMPOOL_ATTR_USED_MEM_HIGH))
    GC.gc(true); CUDA.reclaim()
    return p
end

fmt_ms(t) = @sprintf("%.2f ms", 1e3 * t)
fmt_bytes(n) = HAS_GPU ? Base.format_bytes(n) : "n/a"

# ---------------------------------------------------------------------------------------
# ResNet-18, small-image variant (mirrors examples/resnet_tinyimagenet and the model in
# perf/caching_allocator/benchmark.jl).
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

# A 64x64 RGB batch and its one-hot-shaped target, on the device.
make_batch(bs) = (randn(Float32, 64, 64, 3, bs) |> DEVICE,
                  randn(Float32, NCLASSES, bs)  |> DEVICE)

# ---------------------------------------------------------------------------------------
# Part 1 — speedup and memory saving from autocast vs the Float32 baseline
# ---------------------------------------------------------------------------------------

# `nothing` means "no autocast" (the plain Float32 model); a type means `autocast(_, T)`.
const PRECISIONS = [("Float32 (baseline)", nothing), ("Float16", Float16), ("BFloat16", BFloat16)]

function speedup(bs)
    println("\n", "─"^80)
    println("● ResNet-18 autocast speed & memory — batch $bs")
    println("─"^80)
    @printf("  %-20s %13s %15s %15s\n", "precision", "forward", "fwd+bwd", "peak used (bwd)")

    base_fwd = base_step = 0.0
    for (label, T) in PRECISIONS
        model = resnet18() |> DEVICE
        x, y = make_batch(bs)
        # forward-only, and a full gradient step (the realistic training cost)
        fwd  = T === nothing ? (() -> model(x)) : (() -> autocast(model, T)(x))
        loss = m -> Flux.logitcrossentropy(m(x), y)
        step = T === nothing ? (() -> Flux.gradient(loss, model)) :
                               (() -> Flux.gradient(loss, model; autocast = T))

        t_fwd  = timed(fwd)
        t_step = timed(step; samples = 20)
        mem    = peak_used(step)

        if T === nothing
            base_fwd, base_step = t_fwd, t_step
            @printf("  %-20s %13s %15s %15s\n", label, fmt_ms(t_fwd), fmt_ms(t_step), fmt_bytes(mem))
        else
            @printf("  %-20s %13s %15s %15s   (%.2fx / %.2fx vs baseline)\n", label,
                    fmt_ms(t_fwd), fmt_ms(t_step), fmt_bytes(mem),
                    base_fwd / t_fwd, base_step / t_step)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------------------
# Part 2 — cost of wrapping once vs wrapping each time inside `gradient`
# ---------------------------------------------------------------------------------------

function wrap_overhead(bs; T = BFloat16)
    println("\n", "─"^80)
    println("● Wrapper overhead ($T) — batch $bs")
    println("─"^80)

    model = resnet18() |> DEVICE
    x, y = make_batch(bs)
    loss = m -> Flux.logitcrossentropy(m(x), y)

    # (a) Building the wrapper tree in isolation: a pure-CPU `fmap` walk that allocates the
    #     `AutocastDown`/`AutocastUp` structs. No array is cast here (casts happen in the
    #     forward pass), so this is the raw structural cost paid once per `autocast` call.
    t_build = timed(() -> autocast(model, T))

    # (b) Forward pass — wrap once and reuse, vs wrap on every call.
    wm = autocast(model, T)
    t_fwd_once = timed(() -> wm(x))
    t_fwd_each = timed(() -> autocast(model, T)(x))

    # (c) Gradient step — pre-wrapped model (wrap once) vs the `autocast=T` keyword, which
    #     rebuilds AND back-propagates through the wrapper tree on every call. The keyword
    #     grad is shaped like the original model; the pre-wrapped grad is wrapper-shaped —
    #     the compute is identical, only the wrapper handling differs, which is the point.
    loss_w = m -> Flux.logitcrossentropy(m(x), y)
    t_grad_once = timed(() -> Flux.gradient(loss_w, wm); samples = 20)
    t_grad_each = timed(() -> Flux.gradient(loss, model; autocast = T); samples = 20)

    @printf("  %-42s %13s\n", "autocast(model, T) construction (CPU):", fmt_ms(t_build))
    println()
    @printf("  %-42s %13s\n", "forward, wrap once (reuse):", fmt_ms(t_fwd_once))
    @printf("  %-42s %13s   (+%s)\n", "forward, wrap each call:", fmt_ms(t_fwd_each),
            fmt_ms(t_fwd_each - t_fwd_once))
    println()
    @printf("  %-42s %13s\n", "gradient, wrap once (reuse):", fmt_ms(t_grad_once))
    @printf("  %-42s %13s   (+%s, %.1f%%)\n", "gradient, wrap each call (autocast= kw):",
            fmt_ms(t_grad_each), fmt_ms(t_grad_each - t_grad_once),
            100 * (t_grad_each - t_grad_once) / t_grad_once)
    return nothing
end

# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------

function main()
    if HAS_GPU
        free, total = CUDA.memory_info()
        @info "Running on GPU" CUDA.name(CUDA.device()) free = Base.format_bytes(free) total = Base.format_bytes(total)
    else
        @info "No functional CUDA GPU found: running on CPU (only timings are meaningful)."
    end

    for bs in (64, 128)
        speedup(bs)
    end
    for bs in (64, 128)
        wrap_overhead(bs)
    end

    println("\nDone.")
    return nothing
end

main()
