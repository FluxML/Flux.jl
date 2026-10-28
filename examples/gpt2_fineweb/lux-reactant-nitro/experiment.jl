# The GPT-2 experiment for ReactantNitro: the model, the data and what to do with them. A Nitro
# experiment is a struct plus hooks; the training loop, the compiled programs, the schedules and the
# checkpoints are Nitro's. train.jl runs it, and hellaswag.jl scores it.

using PythonCall   # before Lux, see ../flux-reactant/train.jl

using Random
using Lux, ReactantNitro, Optimisers
using Reactant

include("model.jl")
include("../flux-reactant/data.jl")

# The experiment's fields are all Host values (driver-only), but `half`, which selects the graph.
#
# An epoch is the interval between two validations: `eval_every` optimizer steps, or the whole run
# if it is 0, and Nitro validates, checkpoints and logs at the end of each. The training data is
# read consecutively, as in the other versions: each epoch takes the next window of the token stream.
@experiment struct GPT2
    cfg::NamedTuple = GPT2_TINY
    dataset::String = "tinystories"
    data_dir::String = joinpath(@__DIR__, "..", "data")
    log_dir::String = joinpath(@__DIR__, "log")
    batchsize::Int = 32            # sequences per micro-batch
    total_batch::Int = 2^16        # tokens per optimizer step, reached by accumulating gradients
    steps::Int = 7232              # optimizer steps, a multiple of the epoch length
    eval_every::Int = 250
    val_steps::Int = 20
    hellaswag_every::Int = 0
    ckpt_every::Int = 0
    lr::Float32 = 1f-3
    warmup::Int = 100
    weight_decay::Float32 = 0.1f0
    gradient_clip_norm::Float32 = 1f0
    seed::Int = 0
    start_epoch::Int = 0           # the epochs of a run being resumed
    half::GraphConst{Bool} = true  # bfloat16 matrix multiplications, see `bf16_params`
end

# The `fineweb` preset follows build-nanogpt's GPT-2 (124M) run: 2^19 tokens per step for 19073
# steps, one epoch of FineWeb-Edu's 10B-token sample. `tinystories` is a small model trained for
# one epoch of TinyStories (~474M tokens), which takes ~30 minutes on a single GPU.
ReactantNitro.presets(::Type{GPT2}) = (;
    fineweb = (; cfg=GPT2_SMALL, dataset="fineweb", total_batch=2^19, batchsize=16, lr=6f-4,
                 warmup=715, steps=19073, eval_every=250, val_steps=20, hellaswag_every=250,
                 ckpt_every=5000),
    tinystories = (; cfg=GPT2_TINY, dataset="tinystories", total_batch=2^16, batchsize=32,
                     lr=1f-3, warmup=100, steps=7232, eval_every=250, val_steps=20,
                     hellaswag_every=0, ckpt_every=2000),
)

epoch_steps(e::GPT2) = e.eval_every > 0 ? min(e.eval_every, e.steps) : e.steps

ReactantNitro.max_epochs(e::GPT2) = e.steps ÷ epoch_steps(e)
ReactantNitro.accum(e::GPT2) = e.total_batch ÷ (e.batchsize * e.cfg.block_size)
ReactantNitro.run_dir(e::GPT2) = e.log_dir

# ------------------------------------------------------------------------------------------------
# Data

# The training split: the micro-batches of an epoch, then those of the next epoch, and so on over
# the token stream, wrapping around at its end. Nitro draws one batch to learn the schema before
# training, so the first two `begin_epoch!` calls are for the same epoch.
mutable struct TokenStream{D}
    ds::D
    batchsize::Int
    nbatches::Int      # per epoch
    epoch::Int         # of the first window, 0-based
    calls::Int         # of begin_epoch!
end

Base.length(s::TokenStream) = s.nbatches

ReactantNitro.begin_epoch!(s::TokenStream) = (s.calls += 1; nothing)

function Base.iterate(s::TokenStream, i=(begin_epoch!(s); 1))
    i > s.nbatches && return nothing
    row = ((s.epoch + max(s.calls - 2, 0)) * s.nbatches + i - 1) * s.batchsize + 1
    (x, y), _ = getbatch(s.ds, (row - 1) % length(s.ds) + 1, s.batchsize)
    return (; x, y), i + 1
end

# The datasets are read in the training process, by a single producer, as `datasets` reads through
# Python.
function ReactantNitro.build_data(e::GPT2, dist)
    T, B = e.cfg.block_size, e.batchsize
    train_data = token_blocks(e.dataset, :train; block_size=T, data_dir=e.data_dir)
    train = PrefetchIterator(TokenStream(train_data, B, epoch_steps(e) * ReactantNitro.accum(e),
                                         e.start_epoch, 0); workers=1)
    e.eval_every > 0 || return (; train)
    val_data = token_blocks(e.dataset, :val; block_size=T, data_dir=e.data_dir)
    val = [(; zip((:x, :y), getbatch(val_data, (i - 1) * B + 1, B)[1])...) for i in 1:e.val_steps]
    return (; train, val)
end

# ------------------------------------------------------------------------------------------------
# Model, forward, loss

function ReactantNitro.build_model(e::GPT2, rng)
    model = GPT(e.cfg)
    ps, st = Lux.setup(rng, model)
    return (model, ps, st)
end

# The logits, shape (vocab_size, T, B). With `half`, the model sees a bfloat16 copy of the
# parameters, and the Float32 ones are what the optimizer updates.
function ReactantNitro.forward(e::GPT2, model, ps, st; x)
    return Lux.apply(model, x, e.half ? bf16_params(ps) : ps, st)
end

# The cross-entropy of the next-token prediction, averaged over all positions.
ReactantNitro.loss(::GPT2, logits; y) = sum(token_nll(logits, y)) / length(y)

# The validation loss, the mean of the batches' losses
ReactantNitro.eval_loss(::GPT2) = true

# ------------------------------------------------------------------------------------------------
# Optimization: AdamW with β = (0.9, 0.95) and a decay of 0.1 for matrices only (Nitro does not
# decay the 1-d parameters: biases and LayerNorm), gradient clipping, and linear warmup and cosine
# decay of the learning rate. Nitro's optimizer steps are numbered from 1.

# AdamW is Adam with Nitro's decoupled `Decay` at the end of the chain. `hp.lambda` is the decay
# already scaled by the learning rate.
function ReactantNitro.optimizer(::GPT2, group::Symbol, hp)
    return Optimisers.OptimiserChain(Optimisers.Adam(hp.eta, (0.9f0, 0.95f0), 1f-8),
                                     Decay(hp.lambda, hp.anchor, hp.no_decay_mask))
end
ReactantNitro.learning_rate(e::GPT2) = e.lr
ReactantNitro.lambda(e::GPT2) = e.weight_decay

# Linear warmup, then cosine decay to a tenth of the peak.
function learning_rate_at(e::GPT2, step)
    step <= e.warmup && return e.lr * step / e.warmup
    progress = min((step - e.warmup) / (e.steps - e.warmup), 1)
    return e.lr / 10 + (e.lr - e.lr / 10) * (1 + cos(π * progress)) / 2
end
ReactantNitro.schedules(e::GPT2) = (; eta = _ -> step -> learning_rate_at(e, step))

# Every `ckpt_every` steps, and the last.
function ReactantNitro.checkpointer(e::GPT2)
    e.ckpt_every > 0 || return nothing
    return Checkpointer(; keep=(Last(1), Every(e.ckpt_every ÷ epoch_steps(e))))
end
