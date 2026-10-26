# The training loop, independent of the device: `train.jl` runs it with Reactant, and
# ../flux-cuda/train.jl with CUDA.jl. Load PythonCall before including this file (see train.jl).

using Printf, Random, Statistics
using ArgParse
using Flux
using Flux: Functors
using JLD2

include("model.jl")
include("data.jl")
include("hellaswag.jl")

# The `fineweb` preset follows build-nanogpt's GPT-2 (124M) run: 2^19 tokens per step for 19073
# steps, one epoch of FineWeb-Edu's 10B-token sample. `tinystories` is a small model trained for
# one epoch of TinyStories (~474M tokens), which takes ~30 minutes on a single GPU.
const PRESETS = Dict(
    "fineweb" => (; model=GPT2_SMALL, dataset="fineweb", total_batch=2^19, batchsize=16,
                    lr=6e-4, warmup=715, steps=19073, weight_decay=0.1,
                    eval_every=250, val_steps=20, hellaswag_every=250, ckpt_every=5000),
    "tinystories" => (; model=GPT2_TINY, dataset="tinystories", total_batch=2^16, batchsize=32,
                        lr=1e-3, warmup=100, steps=7232, weight_decay=0.1,
                        eval_every=250, val_steps=20, hellaswag_every=0, ckpt_every=2000),
)

# Linear warmup, then cosine decay to a tenth of the peak.
function learning_rate(step, p)
    step <= p.warmup && return p.lr * step / p.warmup
    progress = min((step - p.warmup) / (p.steps - p.warmup), 1)
    return p.lr / 10 + (p.lr - p.lr / 10) * (1 + cos(π * progress)) / 2
end

# Numerical precision. `bf16`: the parameters and activations are bfloat16, the optimizer updates
# Float32 copies of the parameters (`MixedPrecision`), and the loss is computed in Float32.
# `tf32`: Float32, with matrix multiplications on TF32 tensor cores. `f32`: exact Float32.
# The backend sets the precision of the matrix multiplications for `tf32` and `f32`.
const PRECISIONS = ("bf16", "tf32", "f32")

# Weight decay applies to matrices only, not to biases and LayerNorm parameters.
function no_decay!(opt_state, model)
    for (kp, p) in Flux.trainables(model; path=true)
        ndims(p) == 1 && Flux.adjust!(Functors.getkeypath(opt_state, kp); lambda=0.0)
    end
    return opt_state
end

# FLOPs of a training step per token: 6 per parameter used in a matmul (forward + backward) plus
# the attention scores. The position embedding is a lookup and costs nothing.
function flops_per_token(model, cfg)
    n = sum(length, Flux.trainables(model)) - length(model.wpe.weight)
    return 6n + 12 * cfg.n_layer * cfg.n_embd * cfg.block_size
end

function save_checkpoint(path, model, opt_state, step, pos, cfg)
    mkpath(dirname(path))
    jldsave(path; model_state=cpu(Flux.state(model)), opt_state=cpu(Flux.state(opt_state)),
            step, pos, cfg)
end

# Trains on device `dev`, logging to `log_dir`. `compile(f, args)` returns the function to call in
# place of `f` on arguments like `args`, for the validation loss and the HellaSwag scoring:
# `Reactant.compile` on Reactant, `f` itself to run eagerly. With `gc = true`, an incremental
# garbage collection after each micro-batch frees the memory of the arrays it left behind. The
# other keywords (`batchsize`, `steps`, `lr`, `eval_every`, `hellaswag_every`, `ckpt_every`)
# override the preset's values.
function train(dev; log_dir, compile=(f, args) -> f, gc=false, preset="tinystories",
               data_dir=joinpath(@__DIR__, "..", "data"), precision="bf16", resume=nothing,
               seed=0, peak_tflops=nothing, kws...)
    p = merge(PRESETS[preset], (; (k => v for (k, v) in kws if v !== nothing)...))
    cfg = p.model
    B, T = p.batchsize, cfg.block_size
    p.total_batch % (B * T) == 0 ||
        throw(ArgumentError("total_batch = $(p.total_batch) must be a multiple of batchsize × block_size = $(B * T)"))
    accum = p.total_batch ÷ (B * T)
    precision in PRECISIONS || throw(ArgumentError("precision must be one of $PRECISIONS"))

    Random.seed!(seed)
    train_data = token_blocks(p.dataset, :train; block_size=T, data_dir)
    val_data = token_blocks(p.dataset, :val; block_size=T, data_dir)
    tok = gpt2_tokenizer()
    prompt = tokenize(tok, "Hello, I'm a language model,")
    hellaswag = p.hellaswag_every > 0 ? hellaswag_data(tok) : nothing

    model = GPT(cfg)
    rule = OptimiserChain(AccumGrad(accum), ClipNorm(1.0),
                          AdamW(; eta=p.lr, beta=(0.9, 0.95), lambda=p.weight_decay))
    if precision == "bf16"
        rule = MixedPrecision(Float32, rule)
    end
    step0, pos = 0, 1
    if resume !== nothing
        ckpt = JLD2.load(resume)
        Flux.loadmodel!(model, ckpt["model_state"])
        step0, pos = ckpt["step"], ckpt["pos"]
    end
    flops = flops_per_token(model, cfg)
    precision == "bf16" && (model = Flux.bf16(model))
    model = model |> dev
    opt_state = no_decay!(Flux.setup(rule, model), model)
    resume !== nothing && Flux.loadmodel!(opt_state, ckpt["opt_state"])

    mkpath(log_dir)
    logfile = open(joinpath(log_dir, "log.txt"), "a")
    function logmsg(msg)
        println(msg)
        println(logfile, msg)
        flush(stdout)
        flush(logfile)
    end
    logmsg(@sprintf("[setup] preset=%s params=%.1fM batch=%d×%d accum=%d tokens/step=%d precision=%s steps=%d",
                 preset, sum(length, Flux.trainables(model))/1e6, B, T, accum, p.total_batch,
                 precision, p.steps))

    val_batches = [dev(getbatch(val_data, (i - 1) * B + 1, B)[1]) for i in 1:p.val_steps]
    val_loss = compile(lm_loss, (model, val_batches[1]...))
    eval_hellaswag = hellaswag === nothing ? nothing : hellaswag_evaluator(model, hellaswag; device=dev, compile)

    for step in step0+1:p.steps
        t0 = time()
        lr = learning_rate(step, p)
        Flux.adjust!(opt_state, lr)
        loss = 0.0
        for _ in 1:accum
            batch, pos = getbatch(train_data, pos, B)
            loss += Flux.trainstep!(lm_loss, model, dev(batch), opt_state)
            gc && GC.gc(false)
        end
        loss /= accum
        dt = time() - t0
        tps = p.total_batch / dt
        tflops = flops * tps / 1e12
        mfu = peak_tflops === nothing ? "" : @sprintf(" mfu=%.1f%%", 100tflops / peak_tflops)
        logmsg(@sprintf("[train] step=%d loss=%.4f lr=%.2e dt=%.2fs tok/s=%.0f tflops=%.1f%s",
                     step, loss, lr, dt, tps, tflops, mfu))
        isfinite(loss) || error("non-finite loss at step $step")

        last = step == p.steps
        if p.eval_every > 0 && (step % p.eval_every == 0 || last)
            vl = mean(Float32(val_loss(model, x, y)) for (x, y) in val_batches)
            logmsg(@sprintf("[eval] step=%d val_loss=%.4f", step, vl))
            for (i, s) in enumerate(generate(Flux.f32(cpu(model)), prompt))
                logmsg("[sample $i] " * repr(detokenize(tok, s)))
            end
        end
        if eval_hellaswag !== nothing && (step % p.hellaswag_every == 0 || last)
            (; acc, acc_norm) = eval_hellaswag(model)
            logmsg(@sprintf("[hellaswag] step=%d acc=%.4f acc_norm=%.4f", step, acc, acc_norm))
        end
        if p.ckpt_every > 0 && (step % p.ckpt_every == 0 || last)
            path = joinpath(log_dir, @sprintf("checkpoint_%05d.jld2", step))
            save_checkpoint(path, model, opt_state, step, pos, cfg)
            logmsg("[checkpoint] $path")
        end
    end
    close(logfile)
    return model
end

# The command-line options, without those left unset.
function parse_cli(args; description)
    s = ArgParseSettings(; description, autofix_names=true)
    @add_arg_table! s begin
        "--preset"
            help = "fineweb (GPT-2 124M on FineWeb-Edu 10B) or tinystories (small model, ~30 min)"
            default = "tinystories"
        "--data-dir"
            help = "directory for the tokenized datasets, written on the first run"
            default = normpath(joinpath(@__DIR__, "..", "data"))
        "--log-dir"
            help = "directory for log.txt and checkpoints (default: log/ next to the script)"
        "--batchsize"
            help = "sequences per micro-batch; gradients are accumulated up to the preset's tokens per step"
            arg_type = Int
        "--steps"
            help = "number of optimizer steps (the learning-rate schedule spans them)"
            arg_type = Int
        "--lr"
            help = "peak learning rate"
            arg_type = Float64
        "--eval-every"
            help = "steps between validation loss + samples (0 disables)"
            arg_type = Int
        "--hellaswag-every"
            help = "steps between HellaSwag evaluations (0 disables)"
            arg_type = Int
        "--ckpt-every"
            help = "steps between checkpoints (0 disables)"
            arg_type = Int
        "--precision"
            help = "bf16 (bfloat16 model, Float32 master weights), tf32, or f32"
            default = "bf16"
        "--resume"
            help = "checkpoint to resume from"
        "--seed"
            help = "random seed of the initialization"
            arg_type = Int
            default = 0
        "--peak-tflops"
            help = "peak matmul TFLOPS of the GPU, to report the model FLOPs utilization"
            arg_type = Float64
    end
    return (; (k => v for (k, v) in parse_args(args, s; as_symbols=true) if v !== nothing)...)
end
