# The data pipeline, the model and the training loop, independent of the device:
# flux-cuda/train.jl runs them with CUDA.jl, and flux-reactant/train.jl with Reactant.

# Before Flux: CondaPkg resolves the Python environment when PythonCall loads, capping libstdc++ at
# Julia's, which Julia 1.13 fails to detect once CompilerSupportLibraries_jll is loaded
# (JuliaLang/julia#63967). Without the cap, the environment re-resolves whenever the load order changes.
using HuggingFaceDatasets

using Printf, Random, Statistics
using ArgParse
using Flux
using MLUtils: mapobs

# ------------------------------------------------------------------------------------------------
# Data
#
# ImageNet per-channel mean/std (Tiny-ImageNet is an ImageNet subset), shaped for (W, H, C, N)
# broadcasting.
const NCLASSES = 200
const MEAN = reshape(Float32[0.485, 0.456, 0.406], 1, 1, 3, 1)
const STD  = reshape(Float32[0.229, 0.224, 0.225], 1, 1, 3, 1)

# Decode a raw batch to normalized WHCN Float32. Under HuggingFaceDatasets' "julia" format an
# image column is a stacked (C, W, H, N) UInt8 array — channel axis first, from the numpy->Julia
# axis reversal — so permute to Flux's (W, H, C, N) and standardize per channel.
function decode(batch)
    x = Float32.(batch["image"]) ./ 255f0     # (C, W, H, N)
    x = permutedims(x, (2, 3, 1, 4))          # (W, H, C, N) — Flux WHCN layout
    x = (x .- MEAN) ./ STD
    return (x, batch["label"] .+ 1)           # class labels 1:200 (0-based in the dataset)
end

# Standard small-image training augmentation, applied per image: zero-pad by 4 and take a random
# 64x64 crop, then flip horizontally with probability 1/2.
function augment(xy)
    x, y = xy
    W, H, C, N = size(x)
    pad = 4
    out = similar(x)
    padded = zeros(Float32, W + 2pad, H + 2pad, C)
    for n in 1:N
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

function tinyimagenet(split)
    ds = load_dataset("zh-plus/tiny-imagenet"; split)
    # A handful of images are grayscale: decode every image as 3-channel RGB so that the batches
    # always stack to a uniform (C, W, H, N) array.
    return ds.cast_column("image", datasets.Image(mode="RGB"))
end

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

# The cross-entropy of the integer class labels `y`, and the number of correct predictions, which
# `Flux.trainstep!` returns alongside the loss without differentiating it.
function loss_fn(model, x, y)
    ŷ = model(x)
    return Flux.logitcrossentropy(ŷ, y), (; correct = sum(Flux.onecold(ŷ) .== y))
end

# Trains on device `dev`. `compile(f, args)` returns the function to call in place of `f` on
# arguments like `args`, for the evaluation: `Reactant.compile` on Reactant, `f` itself to run
# eagerly. With `gc = true`, an incremental garbage collection after each training step and each
# evaluation batch frees the memory of the arrays it left behind.
function train(dev; compile=(f, args) -> f, gc=false, epochs=30, batchsize=128, lr=1e-3,
               weight_decay=0.0, num_workers=4, seed=0, clip_norm=false)
    Random.seed!(seed)

    # Dropping the last, smaller batch of an epoch keeps the shape of every training batch the
    # same, which a compiled training step needs: a new shape would compile a new one.
    train_loader = Flux.DataLoader(mapobs(train_transform, tinyimagenet("train"));
                                   batchsize, shuffle=true, partial=false, num_workers)
    val_loader = Flux.DataLoader(mapobs(decode, tinyimagenet("valid")); batchsize, num_workers)
    # The device iterators move each batch to the device and free the previous one.
    train_loader, val_loader = dev(train_loader), dev(val_loader)

    model = resnet18() |> dev
    rule = AdamW(; eta=lr, lambda=weight_decay)
    clip_norm && (rule = OptimiserChain(ClipNorm(), rule))   # norm of each gradient array ≤ 10
    opt_state = Flux.setup(rule, model)
    @printf("[setup] device=%s params=%.2fM epochs=%d batchsize=%d lr=%g weight_decay=%g num_workers=%d\n",
            nameof(typeof(dev)), sum(length, Flux.trainables(model)) / 1e6, epochs, batchsize, lr,
            weight_decay, num_workers)

    # One evaluation function per batch shape: the last validation batch is smaller.
    eval_fns = Dict()
    function evaluate(model)
        testmode!(model)
        lsum, correct, n = 0.0, 0, 0
        for (x, y) in val_loader
            f = get!(() -> compile(loss_fn, (model, x, y)), eval_fns, size(x))
            l, stats = cpu(f(model, x, y))
            lsum += l * length(y)
            correct += stats.correct
            n += length(y)
            gc && GC.gc(false)
        end
        return lsum / n, correct / n
    end

    function log_eval(epoch)
        val_loss, val_acc = evaluate(model)
        @printf("[eval] epoch=%d val_loss=%.4f val_acc=%.4f\n", epoch, val_loss, val_acc)
        flush(stdout)
    end

    log_eval(0)
    step = 0
    for epoch in 1:epochs
        η = lr * (1 + cos(π * (epoch - 1) / epochs)) / 2   # cosine annealing towards 0
        Flux.adjust!(opt_state, η)
        trainmode!(model)
        lsum, correct, n = 0.0, 0, 0
        t = @elapsed for (x, y) in train_loader
            l, stats = Flux.trainstep!(loss_fn, model, (x, y), opt_state)
            lsum += l * length(y)
            correct += stats.correct
            n += length(y)
            step += 1
            # Compiling the first step takes long enough for its arrays to reach the old generation,
            # which only a full collection frees.
            gc && GC.gc(step == 1)
        end
        @printf("[train] epoch=%d lr=%.2e loss=%.4f acc=%.4f time=%.1fs img/s=%.0f\n",
                epoch, η, lsum / n, correct / n, t, n / t)
        log_eval(epoch)
    end
    return model
end

# The command-line options; the defaults are those of `train`.
function parse_cli(args; description)
    s = ArgParseSettings(; description)
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
            help = "random seed"
            arg_type = Int
            default = 0
        "--clip-norm"
            help = "clip the gradient norm of each parameter to 10 (wraps AdamW in OptimiserChain(ClipNorm(), …))"
            action = :store_true
    end
    return (; (Symbol(replace(k, "-" => "_")) => v for (k, v) in parse_args(args, s))...)
end
