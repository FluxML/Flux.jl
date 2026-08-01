# Benchmark for the caching allocator used by `Flux.train!`.
#
# Since https://github.com/FluxML/Flux.jl/pull/2665, `Flux.train!` wraps every training
# step in a `GPUArrays.AllocCache` so that the GPU memory allocated during one step is
# reused by the next one. This keeps the memory footprint stable and avoids the
# ever-growing GPU memory usage reported in
# https://github.com/FluxML/Flux.jl/issues/2523 (and the discourse threads / issues
# #828, #302, #736, JuliaGPU/CUDA.jl#137 linked from there).
#
# The caching allocator can occasionally *slow down* training, which is why `train!`
# accepts a `caching_allocator` keyword to turn it off. This script measures the time /
# memory trade-off of the two settings so the effect can be evaluated on real models.
#
# Run with:
#
#     julia --project=perf/caching_allocator perf/caching_allocator/benchmark.jl
#
# On the first run, instantiate the environment:
#
#     julia --project=perf/caching_allocator -e 'using Pkg; Pkg.instantiate()'

using Flux
using CUDA
using GPUArrays: GPUArrays
using Statistics: mean
using Printf: @printf, @sprintf

const HAS_GPU = CUDA.functional()

# ---------------------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------------------

# Bytes currently reserved by the CUDA memory pool (held by the process, whether in use
# or cached for reuse). This is the number that keeps growing in issue #2523.
reserved_bytes() = HAS_GPU ? CUDA.cached_memory() : 0

# Return the pool to a clean state so successive measurements are comparable.
function reset_gpu!()
    if HAS_GPU
        GC.gc(true)
        CUDA.reclaim()
    end
    return nothing
end

fmt_bytes(n) = HAS_GPU ? Base.format_bytes(n) : "n/a"

# ---------------------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------------------

"""
    run_case(name, make_model, make_data; epochs, caching_allocator)

Move `make_model()` and `make_data()` to the GPU (if available) and time `epochs` calls
to `Flux.train!`, tracking the peak GPU memory reserved by the pool. Returns a NamedTuple
with the per-epoch time and the peak reserved memory (relative to the warmed-up baseline).
"""
function run_case(name, make_model, make_data; epochs::Int, caching_allocator::Bool)
    device = HAS_GPU ? gpu : identity
    model = make_model() |> device
    data = [d |> device for d in make_data()]
    opt = Flux.setup(Adam(), model)
    loss(m, x, y) = Flux.mse(m(x), y)

    # Warm up: compile everything and let the pool reach its steady state before measuring.
    Flux.train!(loss, model, data, opt; caching_allocator)
    reset_gpu!()

    baseline = reserved_bytes()
    peak = baseline
    HAS_GPU && CUDA.synchronize()
    t = @elapsed for _ in 1:epochs
        Flux.train!(loss, model, data, opt; caching_allocator)
        peak = max(peak, reserved_bytes())
    end
    HAS_GPU && CUDA.synchronize()

    result = (; time_per_epoch = t / epochs, peak_reserved = peak - baseline)
    reset_gpu!()
    return result
end

function compare_case(name, make_model, make_data; epochs::Int)
    println("\n", "─"^78)
    println("● ", name)
    println("─"^78)
    off = run_case(name, make_model, make_data; epochs, caching_allocator = false)
    on  = run_case(name, make_model, make_data; epochs, caching_allocator = true)

    @printf("  %-22s %14s %18s\n", "caching_allocator", "time/epoch", "peak reserved")
    @printf("  %-22s %14s %18s\n", "false (off)", @sprintf("%.2f ms", 1e3 * off.time_per_epoch), fmt_bytes(off.peak_reserved))
    @printf("  %-22s %14s %18s\n", "true  (on)",  @sprintf("%.2f ms", 1e3 * on.time_per_epoch),  fmt_bytes(on.peak_reserved))

    speed = off.time_per_epoch / on.time_per_epoch
    @printf("  → caching allocator is %.2f× %s on time", speed >= 1 ? speed : 1/speed,
            speed >= 1 ? "faster" : "slower")
    if HAS_GPU && off.peak_reserved > 0
        @printf(", peak reserved memory %.2f× of off", on.peak_reserved / off.peak_reserved)
    end
    println()
    return (; name, off, on)
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
# Issue #2523 reproduction (forward pass only)
# ---------------------------------------------------------------------------------------

"""
    reproduce_2523()

Reproduce the memory-growth observation from issue #2523: repeatedly run the forward pass
of a `Dense` layer and print the pool usage after each iteration. Without the caching
allocator the reserved memory keeps creeping up.
"""
function reproduce_2523(; num_iters = 10)
    if !HAS_GPU
        @info "No functional CUDA GPU; skipping issue #2523 reproduction."
        return
    end
    println("\n", "─"^78)
    println("● Issue #2523 reproduction: forward-pass memory growth")
    println("─"^78)
    model = Dense(128 => 128) |> gpu
    x = randn(Float32, 128, 128) |> gpu
    reset_gpu!()
    for iter in 1:num_iters
        ŷ = model(x)
        free, _ = CUDA.memory_info()
        @printf("  iter %2d   pool reserved: %10s   free: %10s\n",
                iter, Base.format_bytes(CUDA.cached_memory()), Base.format_bytes(free))
    end
    reset_gpu!()
    return nothing
end

# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------

function main()
    if HAS_GPU
        @info "Running on GPU" CUDA.name(CUDA.device())
    else
        @info "No functional CUDA GPU found: running on CPU (only timings are meaningful)."
    end

    reproduce_2523()

    compare_case("MLP (issue #2523 shapes)", mlp_2523, data_2523; epochs = 20)
    compare_case("Deep MLP, batch 256", mlp_deep, data_deep; epochs = 20)
    compare_case("Small CNN, batch 64", cnn, data_cnn; epochs = 20)
    compare_case("Tiny model (allocator overhead)", tiny, data_tiny; epochs = 20)

    println("\nDone.")
    return nothing
end

main()
