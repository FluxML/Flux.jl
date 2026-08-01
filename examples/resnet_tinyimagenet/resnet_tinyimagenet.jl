# ResNet-18 (small-image variant) on Tiny-ImageNet-200, with Flux + HuggingFaceDatasets.jl.
#
# Tiny-ImageNet is a 200-class subset of ImageNet downsized to 64x64: 100k training and 10k
# validation images. We pull it straight from the HuggingFace Hub through HuggingFaceDatasets.jl,
# augment on the fly, and train a small residual network on the GPU with `Flux.train!`.
#
# Run it with the example's own project, e.g.
#
#     julia --project=. -t auto resnet_tinyimagenet.jl            # 30 epochs (the default)
#     EPOCHS=5 julia --project=. -t auto resnet_tinyimagenet.jl   # override the epoch count
#
# See the README for the config knobs and expected accuracy.

using Random, Statistics
using Flux
using Flux.Losses: logitcrossentropy
using Flux: onehotbatch, onecold, trainmode!, testmode!
using MLUtils: MLUtils, mapobs
using HuggingFaceDatasets
using CUDA, cuDNN

# Train on the GPU when one is available (this example is written for it), else fall back to CPU.
const DEVICE = CUDA.functional() ? gpu : cpu
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
# axis reversal — so permute to Flux's (W, H, C, N) and standardize per channel. Returns a plain
# `(x, y)` tuple: `Flux.train!` splats tuples into the loss, and `for (x, y) in loader` destructures
# them.
function decode(batch)
    x = Float32.(batch["image"]) ./ 255f0     # (C, W, H, N)
    x = permutedims(x, (2, 3, 1, 4))          # (W, H, C, N) — Flux WHCN layout
    x = (x .- MEAN) ./ STD
    return (x, batch["label"])                # labels are 0-based class ids (0:199)
end

# Standard small-image training augmentation, applied per batch: zero-pad by 4 and take a random
# 64x64 crop, then flip horizontally with probability 1/2. Pure Julia (no Python), so it
# parallelizes across worker processes (`num_workers`). Random per call, so it runs every epoch.
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

# Cross-entropy loss and top-1 accuracy over a loader. Switches the model to `testmode!` so
# BatchNorm uses its running statistics (not the per-batch statistics of training).
function loss_and_accuracy(loader, model)
    testmode!(model)
    correct, total = 0, 0
    lsum = 0f0
    for (x, y) in loader
        ŷ = model(x |> DEVICE) |> cpu
        yoh = onehotbatch(y, 0:NCLASSES-1)
        lsum += logitcrossentropy(ŷ, yoh; agg=sum)
        correct += sum(onecold(ŷ, 0:NCLASSES-1) .== y)
        total += length(y)
    end
    return lsum / total, correct / total
end

# `num_workers = 0` loads on the main process; `num_workers > 0` spreads each batch's `getobs` (the
# CPython image decode) over that many worker processes, sidestepping the GIL — the collated batch
# returns through shared memory (MLUtils >= 0.4.12). Pair with `julia -t auto` for threaded collation.
function main(; epochs=30, batchsize=128, lr=1e-3, num_workers=4)
    @info "Setup" DEVICE epochs batchsize lr num_workers

    train_ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    val_ds   = load_dataset("zh-plus/tiny-imagenet", split="valid")
    # A handful of Tiny-ImageNet images are grayscale; force every image to decode as 3-channel RGB
    # so the batch always stacks to a uniform (C, W, H, N) array.
    train_ds = train_ds.cast_column("image", datasets.Image(mode="RGB"))
    val_ds   = val_ds.cast_column("image", datasets.Image(mode="RGB"))

    # Decode + augment on the fly every batch; the CPython decode runs under the GIL, so use
    # `num_workers > 0` to parallelize it across processes.
    train_data = mapobs(train_transform, train_ds)
    val_data   = mapobs(decode, val_ds)

    train_loader = Flux.DataLoader(train_data; batchsize, shuffle=true, num_workers)
    val_loader   = Flux.DataLoader(val_data; batchsize, num_workers)

    model = resnet18() |> DEVICE
    opt = Flux.setup(AdamW(lr), model)

    r(x) = round(x, digits=4)
    r(x::Integer) = x
    for epoch in 0:epochs
        if epoch > 0
            # `train!` puts the model in `trainmode!`. `caching_allocator=false` is needed here:
            # by default `train!` wraps each step in a cross-step allocation cache that keeps every
            # allocation of a step alive for reuse, so the step's peak becomes the *sum* of its
            # allocations (~2x the true working set) — and the cold first step's retained cuDNN
            # algorithm-search workspaces then balloon it past GPU memory and OOM. Disabling the
            # cache restores in-step recycling and keeps peak at the true working set.
            Flux.train!(model, train_loader, opt; caching_allocator=false) do m, x, y
                logitcrossentropy(m(x |> DEVICE), onehotbatch(y, 0:NCLASSES-1) |> DEVICE)
            end
        end
        train_loss, train_acc = loss_and_accuracy(train_loader, model)
        val_loss, val_acc = loss_and_accuracy(val_loader, model)
        @info map(r, (; epoch, train_loss, train_acc, val_loss, val_acc))
    end

    MLUtils.close_dataloader_pool()
    return model
end

function (@main)(args)
    main(; epochs=parse(Int, get(ENV, "EPOCHS", "30")))
end
