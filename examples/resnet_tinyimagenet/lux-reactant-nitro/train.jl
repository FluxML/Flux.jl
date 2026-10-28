# Train the ResNet-18 with Lux and ReactantNitro. The experiment is a struct plus hooks; Nitro
# compiles the training step (forward, backward and optimizer update), the evaluation and the
# data transfers with Reactant (XLA). The data pipeline is in ../data.jl.
#
# Run it with `julia --threads=1,0`: with an interactive thread, Nitro's data tasks run off the main
# thread, from which PythonCall, which reads the dataset, segfaults.

# Read when Reactant initializes: hides the INFO logs of XLA's autotuning.
get!(ENV, "TF_CPP_MIN_LOG_LEVEL", "1")

include("../data.jl")   # loads HuggingFaceDatasets before Lux, see there

using Printf
using ArgParse
using Lux, ReactantNitro, Optimisers
using MLUtils: DataLoader, mapobs

# ------------------------------------------------------------------------------------------------
# Model: the ResNet-18 of ../flux-reactant/training.jl, adapted for small (64x64) images: a single
# 3x3 stride-1 stem instead of the 7x7 stride-2 conv and max-pool, then the four stock stages of
# two BasicBlocks (64->128->256->512), global average pooling and a linear classifier.

# A residual BasicBlock: two 3x3 convs with BatchNorm, plus a skip connection that is a 1x1
# projection conv when the block changes width or stride, and the identity otherwise.
function basic_block(inplanes, planes; stride=1)
    convs = Chain(
        Conv((3, 3), inplanes => planes; stride, pad=1, use_bias=false),
        BatchNorm(planes, relu),
        Conv((3, 3), planes => planes; pad=1, use_bias=false),
        BatchNorm(planes),
    )
    shortcut = if stride != 1 || inplanes != planes
        Chain(Conv((1, 1), inplanes => planes; stride, use_bias=false), BatchNorm(planes))
    else
        NoOpLayer()
    end
    return Chain(Parallel(+, convs, shortcut), WrappedFunction(Base.Fix1(broadcast, relu)))
end

resnet_stage(inplanes, planes, nblocks; stride) =
    Chain(basic_block(inplanes, planes; stride), (basic_block(planes, planes) for _ in 2:nblocks)...)

resnet18(; nclasses=NCLASSES) = Chain(
    Conv((3, 3), 3 => 64; pad=1, use_bias=false),   # small-image stem, no max-pool
    BatchNorm(64, relu),
    resnet_stage(64, 64, 2; stride=1),     # 64x64
    resnet_stage(64, 128, 2; stride=2),    # 32x32
    resnet_stage(128, 256, 2; stride=2),   # 16x16
    resnet_stage(256, 512, 2; stride=2),   #  8x8
    AdaptiveMeanPool((1, 1)),
    FlattenLayer(),                        # (512, N)
    Dense(512 => nclasses),
)

# ------------------------------------------------------------------------------------------------
# The experiment. The fields are Host values: none of them reaches a traced function. The ones
# named like a Nitro run knob (`max_epochs`, `seed`, `gradient_clip_norm`) are read by it directly.

@experiment struct ResNet18
    max_epochs::Int = 30
    batchsize::Int = 128
    lr::Float32 = 1f-3
    weight_decay::Float32 = 0f0
    num_workers::Int = 4
    seed::Int = 0
    gradient_clip_norm::Float32 = 0f0
end

function ReactantNitro.build_model(::ResNet18, rng)
    model = resnet18()
    ps, st = Lux.setup(rng, model)
    return (model, ps, st)
end

# The batches are NamedTuples whose fields the hooks ask for, with the batch dimension last. Named
# functions, not closures, so that the `num_workers` path can ship them to the worker processes.
function train_batch(batch)
    img, label = augment(decode(batch))
    return (; img, label)
end

function eval_batch(batch)
    img, label = decode(batch)
    return (; img, label)
end

# The training split drops its last, smaller batch, so that every training step has the same shape;
# the evaluation split keeps it, and Nitro pads it and slices the outputs back. Loading in worker
# processes delivers the batches as they are ready, which Nitro asks the split to say.
function ReactantNitro.build_data(e::ResNet18, dist)
    loader(split, f; kws...) = DataLoader(mapobs(f, tinyimagenet(split)); e.batchsize, e.num_workers, kws...)
    wrap(l) = e.num_workers > 0 ? PrefetchIterator(l; ordered=false) : l
    return (;
        train = wrap(loader("train", train_batch; shuffle=true, partial=false)),
        val = wrap(loader("valid", eval_batch)),
    )
end

ReactantNitro.forward(::ResNet18, model, ps, st; img) = Lux.apply(model, img, ps, st)

onehot(label) = reshape(1:NCLASSES, :, 1) .== reshape(label, 1, :)

# Cross-entropy of the integer class labels `label` (1:200) given the logits.
ReactantNitro.loss(::ResNet18, logits; label) =
    -sum(onehot(label) .* logsoftmax(logits; dims=1)) / length(label)

# The predictions that give the true class the largest logit.
correct(logits, label) = sum(onehot(label) .* logits; dims=1) .>= maximum(logits; dims=1)

# The validation accuracy is a count over the split, divided once at the end.
ReactantNitro.metrics(::ResNet18, logits; label) = (; acc = sum(correct(logits, label)) => length(label))
ReactantNitro.train_metrics(::ResNet18, logits; label) = (; acc = sum(Float32.(correct(logits, label))) / length(label))
ReactantNitro.eval_loss(::ResNet18) = true

# Adam with decoupled weight decay (`lambda`), as AdamW, and a cosine-annealed learning rate that
# changes once per epoch, as in the other versions: Nitro's schedules are called with the optimizer
# step `t` in 1:total, `total` being the number of steps of the whole run. Nitro does not decay
# biases and normalization parameters.
ReactantNitro.optimizer(::ResNet18) = Optimisers.Adam
ReactantNitro.learning_rate(e::ResNet18) = e.lr
ReactantNitro.lambda(e::ResNet18) = e.weight_decay
function ReactantNitro.schedules(e::ResNet18)
    cosine(total) = t -> e.lr * (1 + cos(π * ((t - 1) ÷ (total ÷ e.max_epochs)) / e.max_epochs)) / 2
    return (; eta = cosine)
end

ReactantNitro.checkpointer(::ResNet18) = nothing

# ------------------------------------------------------------------------------------------------
# The log lines, the same as in the other versions. A Nitro logger is any type with the verbs of
# the logging contract: `log_metrics!` gets the train lines of every optimizer step, the data line
# that ends an epoch's training, and the validation line.

mutable struct ConsoleLogger
    batchsize::Int
    t0::Float64      # when the current epoch's training started
    steps::Int
    loss::Float64
    acc::Float64
    lr::Float64
end

ConsoleLogger(batchsize) = ConsoleLogger(batchsize, time(), 0, 0.0, 0.0, NaN)

function ReactantNitro.log_metrics!(l::ConsoleLogger, metrics; context, epoch, step, kwargs...)
    if context == "train"
        l.steps += 1
        l.loss += metrics.loss
        l.acc += metrics.acc
        l.lr = metrics[Symbol("schedule/default/eta")]
    elseif context == "data"
        t = time() - l.t0
        n = l.steps * l.batchsize
        @printf("[train] epoch=%d lr=%.2e loss=%.4f acc=%.4f time=%.1fs img/s=%.0f\n",
                epoch, l.lr, l.loss / l.steps, l.acc / l.steps, t, n / t)
    elseif context == "validate"
        @printf("[eval] epoch=%d val_loss=%.4f val_acc=%.4f\n", epoch, metrics.loss, metrics.acc)
        l.t0, l.steps, l.loss, l.acc = time(), 0, 0.0, 0.0
    end
    flush(stdout)
    return nothing
end

ReactantNitro.log_params!(::ConsoleLogger, params) = nothing
ReactantNitro.log_tags!(::ConsoleLogger, tags) = nothing
ReactantNitro.log_other!(::ConsoleLogger, key, value) = nothing
ReactantNitro.log_confusion!(::ConsoleLogger, matrix, labels; kwargs...) = nothing
ReactantNitro.finish!(::ConsoleLogger, status) = nothing

# ------------------------------------------------------------------------------------------------

function main(; epochs=30, batchsize=128, lr=1e-3, weight_decay=0.0, num_workers=4, seed=0,
              clip_norm=false)
    e = ResNet18(; max_epochs=epochs, batchsize, lr, weight_decay, num_workers, seed,
                 gradient_clip_norm=clip_norm ? 10f0 : 0f0)
    logger = ConsoleLogger(batchsize)
    nitro = Nitro(e; logger)
    @printf("[setup] params=%.2fM epochs=%d batchsize=%d lr=%g weight_decay=%g num_workers=%d\n",
            Lux.parameterlength(nitro.model) / 1e6, epochs, batchsize, lr, weight_decay, num_workers)
    val = evaluate(nitro; split=:val)
    @printf("[eval] epoch=0 val_loss=%.4f val_acc=%.4f\n", val.loss, val.acc)
    logger.t0 = time()
    train!(nitro)
    return nitro
end

# The command-line options, the same as in the other versions.
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
            help = "clip the global gradient norm to 10"
            action = :store_true
    end
    return (; (Symbol(replace(k, "-" => "_")) => v for (k, v) in parse_args(args, s))...)
end

function (@main)(args)
    main(; parse_cli(args; description="Train a ResNet-18 on Tiny-ImageNet with Lux and ReactantNitro.")...)
    return nothing
end
