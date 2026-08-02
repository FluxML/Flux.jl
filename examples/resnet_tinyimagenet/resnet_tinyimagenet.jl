using Random, Statistics
using ArgParse
using BFloat16s: BFloat16
using Flux
using Flux.Losses: logitcrossentropy
using Flux: onehotbatch, onecold, trainmode!, testmode!
using MLUtils: MLUtils, mapobs
using HuggingFaceDatasets
using CUDA # replace with `Metal` or `AMDGPU` for other backends

const DEVICE = gpu_device() # will select the first available GPU device, or CPU if none are available
const NCLASSES = 200

# ------------------------------------------------------------------------------------------------
# Data
#
# ImageNet per-channel mean/std (Tiny-ImageNet is an ImageNet subset), shaped for (W, H, C, N)
# broadcasting.
const MEAN = reshape(Float32[0.485, 0.456, 0.406], 1, 1, 3, 1)
const STD  = reshape(Float32[0.229, 0.224, 0.225], 1, 1, 3, 1)

# Decode a raw batch to normalized WHCN Float32. Under HuggingFaceDatasets' "julia" format an
# image column is a stacked (C, W, H, N) UInt8 array — channel axis first, from the numpy->Julia
# axis reversal — so permute to Flux's (W, H, C, N) and standardize per channel.
function decode(batch)
    x = Float32.(batch["image"]) ./ 255f0     # (C, W, H, N)
    x = permutedims(x, (2, 3, 1, 4))          # (W, H, C, N) — Flux WHCN layout
    x = (x .- MEAN) ./ STD
    return (x, batch["label"])                # labels are 0-based class ids (0:199)
end

# Standard small-image training augmentation, applied per batch: zero-pad by 4 and take a random
# 64x64 crop, then flip horizontally with probability 1/2.
function augment(xy)
    x, y = xy
    W, H, C, N = size(x)
    pad = 4
    out = similar(x)
    padded = zeros(Float32, W + 2pad, H + 2pad, C)
    for n in 1:N
        fill!(padded, 0f0)
        @views padded[pad+1:pad+W, pad+1:pad+H, :] .= x[:, :, :, n]
        i, j = rand(0:2pad), rand(0:2pad)                  # random top-left crop offset
        if rand(Bool)
            @views out[:, :, :, n] .= padded[i+W:-1:i+1, j+1:j+H, :]   # crop + horizontal flip
        else
            @views out[:, :, :, n] .= padded[i+1:i+W, j+1:j+H, :]      # crop
        end
    end
    return (out, y)
end

# On-the-fly training pipeline: decode then augment. A named function (not a closure) so the
# `num_workers` path can ship it — and the module globals it reads — to worker processes.
train_transform(batch) = augment(decode(batch))

# ------------------------------------------------------------------------------------------------
# Model: ResNet-18 adapted for small (64x64) images.
#
# The textbook ResNet-18 targets 224x224 and downsamples aggressively in the stem (7x7 stride-2
# conv + 3x3 stride-2 max-pool, an /4 before the residual stages), which throws away almost all of
# a 64x64 image. The now-standard fix for small images (CIFAR / Tiny-ImageNet) is a lean stem — a
# single 3x3 stride-1 conv, no max-pool — keeping full resolution into the stages. Everything else
# is stock ResNet-18: four stages of two BasicBlocks each (64->128->256->512 channels), halving the
# spatial size at the start of stages 2-4, then global average pooling and a linear classifier.

# A residual BasicBlock: two 3x3 convs with BatchNorm, plus a skip connection. When the block
# changes width or stride the skip is a 1x1 "projection" conv; otherwise it is the identity.
struct BasicBlock{C, S}
    convs::C
    shortcut::S
end

Flux.@layer BasicBlock

(m::BasicBlock)(x) = relu.(m.convs(x) .+ m.shortcut(x))

function BasicBlock(inplanes::Int, planes::Int; stride::Int=1)
    convs = Chain(
        Conv((3, 3), inplanes => planes; stride, pad=1, bias=false),
        BatchNorm(planes, relu),
        Conv((3, 3), planes => planes; pad=1, bias=false),
        BatchNorm(planes),
    )
    shortcut = if stride != 1 || inplanes != planes
        Chain(Conv((1, 1), inplanes => planes; stride, bias=false), BatchNorm(planes))
    else
        identity
    end
    return BasicBlock(convs, shortcut)
end

# One stage: `nblocks` BasicBlocks, the first optionally downsampling / widening. The blocks have
# different shortcut types (projection Chain vs. `identity`), so collect them untyped and let the
# `Chain` constructor settle the concrete tuple type.
function resnet_stage(inplanes, planes, nblocks; stride)
    blocks = Any[BasicBlock(inplanes, planes; stride)]
    for _ in 2:nblocks
        push!(blocks, BasicBlock(planes, planes))
    end
    return Chain(blocks...)
end

function resnet18(; nclasses=NCLASSES)
    return Chain(
        # small-image stem: 3x3 stride-1, no max-pool (keeps 64x64 resolution)
        Conv((3, 3), 3 => 64; pad=1, bias=false),
        BatchNorm(64, relu),
        resnet_stage(64, 64, 2; stride=1),     # 64x64
        resnet_stage(64, 128, 2; stride=2),    # 32x32
        resnet_stage(128, 256, 2; stride=2),   # 16x16
        resnet_stage(256, 512, 2; stride=2),   #  8x8
        AdaptiveMeanPool((1, 1)),              # global average pool -> 1x1
        Flux.flatten,                          # (512, N)
        Dense(512 => nclasses),
    )
end

# ------------------------------------------------------------------------------------------------
# Training

# Flux ships `f16`/`f32`/`f64` but no `bf16`; this mirrors them (same conversion machinery) so the
# model can be moved to BFloat16 for lower-precision training.
bf16(m) = Flux._paramtype(BFloat16, m)

# Evaluate mean loss and top-1 accuracy over a data loader. `cast` converts a device batch to the
# training precision (identity for Float32, `BFloat16.` for `--bfloat16`); metrics accumulate in
# Float32 regardless of the model's precision.
function loss_and_accuracy(loader, model, cast=identity)
    testmode!(model)
    correct, total = 0, 0
    lsum = 0f0
    for (x, y) in loader
        ŷ = Float32.(model(cast(x |> DEVICE)) |> cpu)
        yoh = onehotbatch(y, 0:NCLASSES-1)
        lsum += logitcrossentropy(ŷ, yoh; agg=sum)
        correct += sum(onecold(ŷ, 0:NCLASSES-1) .== y)
        total += length(y)
    end
    return lsum / total, correct / total
end

function main(; epochs=30, batchsize=128, lr=1e-3, weight_decay=0.0,
              num_workers=4, seed=0, clip_norm=false, bfloat16=false)
    Random.seed!(seed)
    # `cast` moves a device batch to the training precision (a no-op for Float32).
    cast = bfloat16 ? (x -> BFloat16.(x)) : identity
    @info "Setup" DEVICE epochs batchsize lr weight_decay num_workers seed clip_norm bfloat16

    train_ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    val_ds   = load_dataset("zh-plus/tiny-imagenet", split="valid")
    # A handful of Tiny-ImageNet images are grayscale; force every image to decode as 3-channel RGB
    # so the batch always stacks to a uniform (C, W, H, N) array.
    train_ds = train_ds.cast_column("image", datasets.Image(mode="RGB"))
    val_ds   = val_ds.cast_column("image", datasets.Image(mode="RGB"))

    train_data = mapobs(train_transform, train_ds)
    val_data   = mapobs(decode, val_ds)

    train_loader = Flux.DataLoader(train_data; batchsize, shuffle=true, num_workers)
    val_loader   = Flux.DataLoader(val_data; batchsize, num_workers)

    model = resnet18()
    bfloat16 && (model = bf16(model))
    model = model |> DEVICE

    rule = AdamW(; eta=lr, lambda=weight_decay)
    clip_norm && (rule = OptimiserChain(ClipNorm(), rule))   # clip gradient L2 norm, then AdamW step
    opt = Flux.setup(rule, model)

    r(x) = round(x, digits=4)
    r(x::Integer) = x

    # `DEVICE(train_loader)` is a device iterator: it moves each batch to the GPU and frees the
    # previous one, so the CUDA memory pool holds a flat working set instead of growing epoch over
    # epoch (which per-step `x |> DEVICE` allocations would otherwise cause). Labels arrive on the
    # GPU too, so `onehotbatch` runs there.
    for epoch in 0:epochs
        # Per-epoch cosine annealing of the learning rate, from `lr` towards 0 over training
        # (Lux's example schedules per iteration; per epoch keeps the single `Flux.train!` call).
        η = lr * (1 + cos(π * max(epoch - 1, 0) / epochs)) / 2
        t = 0.0
        if epoch > 0
            Flux.adjust!(opt, η)
            t = @elapsed Flux.train!(model, DEVICE(train_loader), opt) do m, x, y
                logitcrossentropy(m(cast(x)), onehotbatch(y, 0:NCLASSES-1))
            end
        end
        train_loss, train_acc = loss_and_accuracy(train_loader, model, cast)
        val_loss, val_acc = loss_and_accuracy(val_loader, model, cast)
        @info map(r, (; epoch, lr=η, train_loss, train_acc, val_loss, val_acc, time=t))
    end

    return model
end

# Command-line interface. The defaults mirror `main`'s keyword defaults and are surfaced in `--help`.
function parse_cli(ARGS)
    s = ArgParseSettings(description="Train a small-image ResNet-18 on Tiny-ImageNet-200.")
    @add_arg_table! s begin
        "--epochs"
            help = "number of training epochs"
            arg_type = Int
            default = 30
        "--batchsize"
            help = "minibatch size"
            arg_type = Int
            default = 128
        "--lr"
            help = "peak AdamW learning rate (cosine-annealed towards 0 over training)"
            arg_type = Float64
            default = 1e-3
        "--weight-decay"
            help = "AdamW decoupled weight decay (0 = plain Adam)"
            arg_type = Float64
            default = 0.0
        "--num-workers"
            help = "data-loading worker processes (0 loads in the main process)"
            arg_type = Int
            default = 4
        "--seed"
            help = "random seed for reproducibility"
            arg_type = Int
            default = 0
        "--clip-norm"
            help = "clip the gradient L2 norm to 10 (wraps AdamW in OptimiserChain(ClipNorm(), …))"
            action = :store_true
        "--bfloat16"
            help = "train in BFloat16 instead of Float32"
            action = :store_true
    end
    return parse_args(ARGS, s)
end

function (@main)(ARGS)
    opts = parse_cli(ARGS)
    main(;
        epochs       = opts["epochs"],
        batchsize    = opts["batchsize"],
        lr           = opts["lr"],
        weight_decay = opts["weight-decay"],
        num_workers  = opts["num-workers"],
        seed         = opts["seed"],
        clip_norm    = opts["clip-norm"],
        bfloat16     = opts["bfloat16"],
    )
    # `main` returns the model; a script entry point must return `nothing` or an integer, else
    # Julia's `Cint(ret)` at exit throws and the process exits non-zero.
    return nothing
end
