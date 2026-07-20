# VGG16 on CIFAR-10 — a performance playground for Flux's step-by-step training API.
#
# This example is adapted from the FluxML model-zoo
# (https://github.com/FluxML/model-zoo/blob/master/vision/vgg_cifar10/vgg_cifar10.jl)
# and instrumented to report, per epoch:
#   * wall-clock training time
#   * GPU memory in use (device-level) and peak observed
#
# It contrasts where a `GPUArrays.AllocCache` (reused across steps) wraps the step, to see
# how the caching granularity affects GPU memory (PR #2681):
#   * `:manual`       — classic `gradient` + `update!`, no cache (model-zoo baseline).
#   * `:cache_both`   — cache gradient + update! together (what `Flux.train_step!` does today).
#   * `:cache_grad`   — cache only the gradient pass; run update! outside the cache.
#   * `:cache_update` — cache only the update!; run the gradient pass outside the cache.
#
# Usage (from this folder):
#   julia --project -e 'include("vgg_cifar10.jl"); main(; method=:cache_grad, epochs=3)'
#   julia --project bench_driver.jl cache_both 128 3 5120   # fresh process per config

using Flux
using Flux: onehotbatch, onecold, logitcrossentropy
using GPUArrays: AllocCache, @cached, unsafe_free!
using Statistics: mean
using CUDA
using MLDatasets: CIFAR10
using MLUtils: splitobs, DataLoader
using Printf

# Accept dataset download prompts non-interactively.
get!(ENV, "DATADEPS_ALWAYS_ACCEPT", "true")

if CUDA.functional()
    @info "CUDA is on" device = CUDA.name(CUDA.device())
    CUDA.allowscalar(false)
end

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

function get_processed_data(args)
    x, y = CIFAR10(:train)[:]

    # Optionally use only a subset of the training set for quick perf runs.
    if args.ntrain > 0 && args.ntrain < size(x, 4)
        x = x[:, :, :, 1:args.ntrain]
        y = y[1:args.ntrain]
    end

    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at = 1 - args.valsplit)

    train_x = float(train_x)
    train_y = onehotbatch(train_y, 0:9)
    val_x = float(val_x)
    val_y = onehotbatch(val_y, 0:9)

    return (train_x, train_y), (val_x, val_y)
end

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

function vgg16()
    Chain(
        Conv((3, 3), 3 => 64, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(64),
        MaxPool((2, 2)),
        Conv((3, 3), 64 => 128, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(128),
        MaxPool((2, 2)),
        Conv((3, 3), 128 => 256, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(256),
        MaxPool((2, 2)),
        Conv((3, 3), 256 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        MaxPool((2, 2)),
        Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = (1, 1)),
        BatchNorm(512),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(512, 4096, relu),
        Dropout(0.5),
        Dense(4096, 4096, relu),
        Dropout(0.5),
        Dense(4096, 10),
    )
end

Base.@kwdef mutable struct Args
    batchsize::Int = 128
    lr::Float32 = 3.0f-4
    epochs::Int = 3
    valsplit::Float64 = 0.1
    ntrain::Int = 0            # 0 = use the whole training set
    # :manual | :cache_both (== Flux.train_step!) | :cache_grad | :cache_update
    method::Symbol = :cache_both
end

# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

# Live GPU bytes currently allocated by CUDA.jl's pool (MiB). This is the actual
# working-set — unlike `total_memory() - free_memory()`, it is not inflated by
# memory the pool has reserved from the driver but is not currently using.
gpu_used_mib() = CUDA.used_memory() / 2^20

# Memory the pool has reserved from the driver (live + cached-but-free), in MiB.
gpu_reserved_mib() = (CUDA.used_memory() + CUDA.cached_memory()) / 2^20

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

loss(m, x, y) = logitcrossentropy(m(x), y)

# A single optimisation step, with the `@cached` region placed at different granularities.
# `cache` is an `AllocCache` reused across steps (ignored by `:manual`).

# no caching (model-zoo baseline)
function step!(::Val{:manual}, model, opt, cache, x, y)
    gs = Flux.gradient(m -> loss(m, x, y), model)
    Flux.update!(opt, model, gs[1])
    return model, opt
end

# cache the whole step: gradient + update! (this is what `Flux.train_step!` does today)
function step!(::Val{:cache_both}, model, opt, cache, x, y)
    @cached cache begin
        gs = Flux.gradient(m -> loss(m, x, y), model)
        Flux.update!(opt, model, gs[1])
    end
    return model, opt
end

# cache only the gradient pass; run update! outside the cache
function step!(::Val{:cache_grad}, model, opt, cache, x, y)
    gs = @cached cache (Flux.gradient(m -> loss(m, x, y), model))
    Flux.update!(opt, model, gs[1])
    return model, opt
end

# cache only the update!; run the gradient pass outside the cache
function step!(::Val{:cache_update}, model, opt, cache, x, y)
    gs = Flux.gradient(m -> loss(m, x, y), model)
    @cached cache (Flux.update!(opt, model, gs[1]))
    return model, opt
end

function train_epoch!(method::Val, model, opt, cache, loader)
    peak = 0.0
    for (x, y) in loader
        x, y = gpu(x), gpu(y)
        model, opt = step!(method, model, opt, cache, x, y)
        peak = max(peak, gpu_used_mib())
    end
    return model, opt, peak
end

function validate(model, loader)
    l = 0.0f0
    for (x, y) in loader
        x, y = gpu(x), gpu(y)
        l += loss(model, x, y)
    end
    return l / length(loader)
end

function main(; kws...)
    args = Args(; kws...)
    @info "Args" args

    train_data, val_data = get_processed_data(args)
    train_loader = DataLoader(train_data, batchsize = args.batchsize, shuffle = true)
    val_loader = DataLoader(val_data, batchsize = args.batchsize)

    @info "Constructing model on GPU"
    model = vgg16() |> gpu

    opt_state = Flux.setup(Adam(args.lr), model)
    cache = AllocCache()
    method = Val(args.method)

    # Warm up (compilation) so the first timed epoch is representative.
    @info "Warming up (compilation)…"
    CUDA.reclaim()
    peak_mib = 0.0
    baseline_mib = gpu_used_mib()

    @info @sprintf("Baseline GPU memory in use: %.1f MiB", baseline_mib)
    @info "Training with method = $(args.method)"

    try
        for epoch in 1:args.epochs
            CUDA.synchronize()
            t0 = time()

            model, opt_state, epoch_peak = train_epoch!(method, model, opt_state, cache, train_loader)

            CUDA.synchronize()
            dt = time() - t0

            peak_mib = max(peak_mib, epoch_peak)
            reserved = gpu_reserved_mib()

            vloss = validate(model, val_loader)

            @info @sprintf(
                "epoch %2d | time %6.2fs | val_loss %.4f | peak_used %7.1f MiB | reserved %7.1f MiB",
                epoch, dt, vloss, epoch_peak, reserved
            )
        end
    catch err
        if err isa OutOfGPUMemoryError
            @warn "OUT OF GPU MEMORY" method = args.method batchsize = args.batchsize peak_mib
            return nothing
        else
            rethrow(err)
        end
    end

    @info @sprintf(
        "Done. method=%s batchsize=%d | peak GPU memory in use: %.1f MiB | cache holds %.1f MiB",
        args.method, args.batchsize, peak_mib, sizeof(cache) / 2^20
    )
    unsafe_free!(cache)
    return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
