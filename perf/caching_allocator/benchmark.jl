# Benchmark for the caching allocator used by `Flux.train!`.
#
# Since https://github.com/FluxML/Flux.jl/pull/2665, `Flux.train!` wraps every training
# step in a `GPUArrays.AllocCache` so that the GPU memory allocated during one step is
# reused by the next one. This keeps *reserved* memory stable and avoids the ever-growing
# GPU memory usage reported in https://github.com/FluxML/Flux.jl/issues/2523.
#
# The cache is not free, though: within a step it *pins* every allocation until the step
# ends, so a step's peak becomes the sum of its allocations rather than its working set.
# For allocation-heavy models (deep conv nets) that inflates memory and can OOM. This script
# compares two memory axes — peak *used* (live bytes, what OOMs) and peak *reserved* (pool
# high-water, what issue #2523 is about) — across a few models and several allocator policies:
#
#   * cache on             — the default `GPUArrays.AllocCache` (with the first-step skip)
#   * cache off            — plain pool + GC (issue #2523 regime: reserved creeps up)
#   * cache off + gc/N     — no cache, but a periodic `GC.gc(false)` every N steps to bound
#                            the reserved-memory growth without a full collection each step
#
# It also reproduces the forward-pass memory growth from #2523, and does a cold-start check
# on ResNet-18 (the first `train!` call, which runs cuDNN's convolution-algorithm search)
# to confirm the first-step cache skip keeps those one-off probe workspaces from being
# pinned.
#
# Peak memory is read from the CUDA pool's exact high-water marks (no sampling needed).
#
# Run with:
#
#     julia --project=perf/caching_allocator perf/caching_allocator/benchmark.jl
#
# On the first run, instantiate the environment:
#
#     julia --project=perf/caching_allocator -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'

using Flux
using CUDA, cuDNN
using GPUArrays: GPUArrays
using Statistics: mean
using Printf: @printf, @sprintf

const HAS_GPU = CUDA.functional()

# ---------------------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------------------

# Bytes currently reserved by the CUDA memory pool (held by the process, whether in use or
# cached for reuse). This is the number that keeps growing in issue #2523.
reserved_bytes() = HAS_GPU ? Int(CUDA.cached_memory()) : 0

# Bytes currently live (allocated from the pool). Its peak over a step is the true working
# set — the number that actually determines an OOM. Always `reserved >= used`; the gap is
# free-but-held memory (dead buffers the GC hasn't reclaimed yet, plus, with the cache on,
# the buffers pinned in the cache's free pool).
used_bytes() = HAS_GPU ? Int(CUDA.used_memory()) : 0

# Return the pool to a clean state so successive measurements are comparable.
function reset_gpu!()
    if HAS_GPU
        GC.gc(true)
        CUDA.reclaim()
    end
    return nothing
end

fmt_bytes(n) = HAS_GPU ? Base.format_bytes(n) : "n/a"

# `used` peaks *inside* a step, but we don't need to sample for it: the CUDA memory pool keeps
# exact high-water marks for both used and reserved bytes. We reset them to the current value
# before a measured run and read them after — this is exact (the driver sees every allocation)
# and needs no sampling thread. `pool_create` returns the very pool CUDA.jl allocates from
# (the one `used_memory()` / `cached_memory()` query).
gpu_pool() = CUDA.CUDACore.pool_create(CUDA.device())

function reset_peaks!()
    HAS_GPU || return nothing
    p = gpu_pool()
    CUDA.attribute!(p, CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
    CUDA.attribute!(p, CUDA.MEMPOOL_ATTR_RESERVED_MEM_HIGH, UInt64(0))
    return nothing
end

peak_used_bytes()     = HAS_GPU ? Int(CUDA.attribute(UInt64, gpu_pool(), CUDA.MEMPOOL_ATTR_USED_MEM_HIGH)) : 0
peak_reserved_bytes() = HAS_GPU ? Int(CUDA.attribute(UInt64, gpu_pool(), CUDA.MEMPOOL_ATTR_RESERVED_MEM_HIGH)) : 0

# ---------------------------------------------------------------------------------------
# Allocator policies to compare
# ---------------------------------------------------------------------------------------

# Explicit `gc_interval` on every config so these are pinned regardless of `train!`'s defaults
# (which are `caching_allocator = false, gc_interval = :auto`).
const CONFIGS = [
    ("cache on",           (; caching_allocator = true,  gc_interval = 0)),
    ("cache off",          (; caching_allocator = false, gc_interval = 0)),
    ("cache off + gc/1",   (; caching_allocator = false, gc_interval = 1)),
    ("cache off + gc/4",   (; caching_allocator = false, gc_interval = 4)),
    ("cache off + gc/auto",(; caching_allocator = false, gc_interval = :auto)),
]

# ---------------------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------------------

"""
    run_case(name, make_model, make_data; epochs, config, warmup=true)

Move `make_model()` and `make_data()` to the GPU (if available) and time `epochs` calls to
`Flux.train!` under the allocator policy `config` (a NamedTuple of `train!` keywords),
tracking the **absolute** peak GPU memory (both live `used` and pool `reserved`) reached
during training. Returns a NamedTuple `(time_per_epoch, peak_used, peak_reserved)`, or
`nothing` if the run runs out of GPU memory. With `warmup=false` the very first (cold)
`train!` call is the one measured — used for the cold-start check.
"""
function run_case(name, make_model, make_data; epochs::Int, config, warmup::Bool = true)
    device = HAS_GPU ? gpu : identity
    model = make_model() |> device
    data = [d |> device for d in make_data()]
    opt = Flux.setup(Adam(), model)
    loss(m, x, y) = Flux.mse(m(x), y)

    try
        # Warm up: compile everything and let the pool reach its steady state before measuring.
        if warmup
            Flux.train!(loss, model, data, opt; config...)
            reset_gpu!()
        end

        HAS_GPU && (CUDA.synchronize(); reset_peaks!())       # zero the pool high-water marks
        t = @elapsed for _ in 1:epochs
            Flux.train!(loss, model, data, opt; config...)
        end
        HAS_GPU && CUDA.synchronize()
        result = (; time_per_epoch = t / epochs,
                    peak_used = peak_used_bytes(), peak_reserved = peak_reserved_bytes())
        reset_gpu!()
        return result
    catch e
        (e isa OutOfGPUMemoryError) || rethrow()
        reset_gpu!()
        return nothing
    end
end

function compare_case(name, make_model, make_data; epochs::Int)
    println("\n", "─"^78)
    println("● ", name)
    println("─"^78)
    @printf("  %-20s %13s %15s %15s\n", "config", "time/epoch", "peak used", "peak reserved")
    for (label, config) in CONFIGS
        r = run_case(name, make_model, make_data; epochs, config)
        if r === nothing
            @printf("  %-20s %13s %15s %15s\n", label, "—", "OOM", "OOM")
        else
            @printf("  %-20s %13s %15s %15s\n", label,
                    @sprintf("%.2f ms", 1e3 * r.time_per_epoch),
                    fmt_bytes(r.peak_used), fmt_bytes(r.peak_reserved))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------------------
# Model / data cases
# ---------------------------------------------------------------------------------------

# The minimal MLP from issue #2523: a single Dense layer, fixed batch size. The forward
# pass alone made the pool grow ~0.5 MiB per iteration.
mlp_2523() = Dense(128 => 128)
data_2523() = [(randn(Float32, 128, 128), randn(Float32, 128, 128)) for _ in 1:50]

# A deeper MLP with a larger batch: bigger working set, so the memory difference between
# the two allocators is more pronounced.
mlp_deep() = Chain(Dense(512 => 1024, relu), Dense(1024 => 1024, relu), Dense(1024 => 512))
data_deep() = [(randn(Float32, 512, 256), randn(Float32, 512, 256)) for _ in 1:50]

# A small conv net (CIFAR-ish shapes) exercising the more allocation-heavy conv path.
cnn() = Chain(
    Conv((3, 3), 3 => 32, relu; pad = 1), MaxPool((2, 2)),
    Conv((3, 3), 32 => 64, relu; pad = 1), MaxPool((2, 2)),
    Flux.flatten, Dense(64 * 8 * 8 => 10),
)
data_cnn() = [(randn(Float32, 32, 32, 3, 64), randn(Float32, 10, 64)) for _ in 1:20]

# A tiny model where each step does almost no compute: the caching allocator's bookkeeping
# (locking, hashing allocation sizes) is then a relatively large fraction of the step, which
# is the regime where turning it off can be faster.
tiny() = Dense(16 => 16)
data_tiny() = [(randn(Float32, 16, 64), randn(Float32, 16, 64)) for _ in 1:200]

# ---------------------------------------------------------------------------------------
# ResNet-18, small-image variant (mirrors examples/resnet_tinyimagenet). This is the
# allocation-heavy, deep conv net where the caching allocator's per-step pinning inflates
# peak memory the most.
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

# Synthetic Tiny-ImageNet-shaped batches (64x64 RGB, 200 classes).
resnet_data(bs; n = 5) = [(randn(Float32, 64, 64, 3, bs), randn(Float32, NCLASSES, bs)) for _ in 1:n]

# ---------------------------------------------------------------------------------------
# Issue #2523 reproduction (forward pass only)
# ---------------------------------------------------------------------------------------

"""
    reproduce_2523()

Reproduce the memory-growth observation from issue #2523: repeatedly run a forward pass and
print the pool status after each iteration. Each output stays live until the GC frees it, and
the GC seldom fires on its own (the `CuArray` wrappers are tiny on the CPU heap), so `used`
(pool usage) creeps up every iteration — and once it crosses the pool's current reservation,
`reserved` climbs too. A per-step `GC.gc(false)` — what `gc_interval=1` does — flattens both.
"""
function reproduce_2523(; num_iters = 20)
    if !HAS_GPU
        @info "No functional CUDA GPU; skipping issue #2523 reproduction."
        return
    end
    println("\n", "─"^78)
    println("● Issue #2523 reproduction: forward-pass memory growth")
    println("─"^78)
    # Dense(1024⇒1024) rather than the issue's Dense(128⇒128) so that `reserved` visibly grows
    # too, not just `used` (identical mechanism; the tiny layer just keeps `used` under the
    # 32 MiB minimum block, so only `used` moves — as in the issue's own output).
    model = Dense(1024 => 1024) |> gpu
    x = randn(Float32, 1024, 1024) |> gpu
    log_row(iter) = (iter <= 3 || iter % 5 == 0) &&
        @printf("    iter %2d   used %10s   reserved %10s\n", iter,
                fmt_bytes(used_bytes()), fmt_bytes(reserved_bytes()))

    println("  no GC (buffers linger — used and reserved creep up):")
    reset_gpu!()
    for iter in 1:num_iters
        ŷ = model(x)
        log_row(iter)
    end

    println("  GC.gc(false) every step (what gc_interval=1 does — stays flat):")
    reset_gpu!()
    for iter in 1:num_iters
        ŷ = model(x)
        GC.gc(false)
        log_row(iter)
    end
    reset_gpu!()
    return nothing
end

# ---------------------------------------------------------------------------------------
# Cold-start check: the first `train!` call runs cuDNN's convolution-algorithm search,
# whose one-off probe workspaces used to be pinned by the cache and blow past GPU memory.
# `train!` now skips the cache on the first step, so a cold run should peak at about the
# same as a warm one (rather than ballooning / OOMing).
# ---------------------------------------------------------------------------------------

function cold_start_check(; batchsizes = (64, 128))
    if !HAS_GPU
        @info "No functional CUDA GPU; skipping cold-start check."
        return
    end
    println("\n", "─"^78)
    println("● ResNet-18 cold-start, cache on (first-step cache skip) — cold vs warm peak")
    println("─"^78)
    @printf("  %-7s %13s %13s %13s %13s\n", "batch",
            "used (cold)", "resv (cold)", "used (warm)", "resv (warm)")
    for bs in batchsizes
        cold = run_case("resnet cold bs$bs", () -> resnet18(), () -> resnet_data(bs; n = 4);
                        epochs = 1, config = (; caching_allocator = true, gc_interval = 0), warmup = false)
        warm = run_case("resnet warm bs$bs", () -> resnet18(), () -> resnet_data(bs; n = 4);
                        epochs = 1, config = (; caching_allocator = true, gc_interval = 0), warmup = true)
        cell(r, f) = r === nothing ? "OOM" : fmt_bytes(f(r))
        @printf("  %-7s %13s %13s %13s %13s\n", string(bs),
                cell(cold, r -> r.peak_used), cell(cold, r -> r.peak_reserved),
                cell(warm, r -> r.peak_used), cell(warm, r -> r.peak_reserved))
    end
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

    reproduce_2523()
    cold_start_check()

    compare_case("MLP (issue #2523 shapes)", mlp_2523, data_2523; epochs = 20)
    compare_case("Deep MLP, batch 256", mlp_deep, data_deep; epochs = 20)
    compare_case("Small CNN, batch 64", cnn, data_cnn; epochs = 20)
    compare_case("Tiny model (allocator overhead)", tiny, data_tiny; epochs = 20)
    compare_case("ResNet-18, batch 64", () -> resnet18(), () -> resnet_data(64); epochs = 5)
    compare_case("ResNet-18, batch 128", () -> resnet18(), () -> resnet_data(128); epochs = 5)

    println("\nDone.")
    return nothing
end

main()
