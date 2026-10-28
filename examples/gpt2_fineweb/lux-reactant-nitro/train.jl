# Pretrain GPT-2 with Lux and ReactantNitro. The experiment is in experiment.jl: Nitro compiles the
# training step (forward, backward, gradient accumulation and optimizer update), the validation and
# the data transfers with Reactant (XLA). The first run for a dataset streams and tokenizes it into
# `data_dir` (see ../flux-reactant/data.jl).
#
# Run it with `julia --threads=1,0`: with an interactive thread, Nitro's data tasks run off the main
# thread, from which PythonCall, which reads the datasets, segfaults.

# Read when Reactant initializes. XLA reserves 75% of the GPU memory by default; the INFO logs it
# would print include a register-spill note for every matmul kernel it autotunes.
get!(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "0.9")
get!(ENV, "TF_CPP_MIN_LOG_LEVEL", "1")

include("experiment.jl")   # loads PythonCall before Lux, see there
include("hellaswag.jl")

using Printf, Statistics
using ArgParse
using Reactant: PrecisionConfig

# ------------------------------------------------------------------------------------------------
# The log lines, the same as in the other versions, printed and appended to `log.txt`. A Nitro
# logger is any type with the verbs of the logging contract: `log_metrics!` gets the line of each
# optimizer step. The loss of a step is that of its last micro-batch.

mutable struct ConsoleLogger
    file::IO
    total_batch::Int
    flops::Float64                         # of a token
    peak_tflops::Union{Float64, Nothing}
    t_last::Float64                        # when the previous step ended
end

function logmsg(l::ConsoleLogger, msg)
    println(msg)
    println(l.file, msg)
    flush(stdout)
    flush(l.file)
end

function ReactantNitro.log_metrics!(l::ConsoleLogger, metrics; context, step, kwargs...)
    if context == "train"
        t = time()
        dt, l.t_last = t - l.t_last, t
        tps = l.total_batch / dt
        tflops = l.flops * tps / 1e12
        mfu = l.peak_tflops === nothing ? "" : @sprintf(" mfu=%.1f%%", 100tflops / l.peak_tflops)
        lr = get(metrics, Symbol("schedule/default/eta"), NaN)
        logmsg(l, @sprintf("[train] step=%d loss=%.4f lr=%.2e dt=%.2fs tok/s=%.0f tflops=%.1f%s",
                           step, metrics.loss, lr, dt, tps, tflops, mfu))
    end
    return nothing
end

ReactantNitro.log_params!(::ConsoleLogger, params) = nothing
ReactantNitro.log_tags!(::ConsoleLogger, tags) = nothing
ReactantNitro.log_other!(::ConsoleLogger, key, value) = nothing
ReactantNitro.log_confusion!(::ConsoleLogger, matrix, labels; kwargs...) = nothing
ReactantNitro.finish!(::ConsoleLogger, status) = nothing

# After each validation, the log lines of the validation loss, samples from the model and HellaSwag,
# as in the other versions. Nitro calls the monitor at every phase transition: the one out of the
# validation has the metrics in `info`, and the one into the training of an epoch after the first
# restarts the clock of the step times, which for the first step includes the compilation.
function monitor_evaluations!(nitro, e::GPT2, logger, tok)
    prompt = tokenize(tok, "Hello, I'm a language model,")
    hellaswag = e.hellaswag_every > 0 ? hellaswag_data(tok) : nothing
    evaluate_hellaswag = nothing
    function monitor(phase, step, epoch, info)
        phase isa TrainStepping && epoch > e.start_epoch + 1 && (logger.t_last = time())
        haskey(info, :metrics) || return nothing
        logmsg(logger, @sprintf("[eval] step=%d val_loss=%.4f", step, info.metrics.loss))
        ps, st = cpu_device()((parameters(nitro), nitro.st))
        for (i, s) in enumerate(generate(nitro.model, ps, st, prompt))
            logmsg(logger, "[sample $i] " * repr(detokenize(tok, s)))
        end
        if hellaswag !== nothing && step % e.hellaswag_every == 0
            if evaluate_hellaswag === nothing
                evaluate_hellaswag = hellaswag_evaluator(nitro.model, nitro.ps, nitro.st, hellaswag; half=e.half)
            end
            (; acc, acc_norm) = evaluate_hellaswag(nitro.ps)
            logmsg(logger, @sprintf("[hellaswag] step=%d acc=%.4f acc_norm=%.4f", step, acc, acc_norm))
        end
        logger.t_last = time()
        return nothing
    end
    register_phase_monitor!(nitro, monitor)
end

# ------------------------------------------------------------------------------------------------

# Numerical precision. `bf16`: the model sees bfloat16 parameters and activations, with Float32
# LayerNorm and loss and Float32 master weights. `tf32`: Float32, with matrix multiplications on TF32
# tensor cores, XLA's default precision for Float32 matrix multiplications on NVIDIA GPUs. `f32`:
# exact Float32. Reactant reads the setting when it compiles a function, so it wraps the run.
const PRECISIONS = ("bf16", "tf32", "f32")
precision_config(precision) = precision == "f32" ? (; dot_general_precision=PrecisionConfig.HIGHEST) : (;)

# The experiment of a preset, the keywords overriding its fields. The number of steps is rounded
# down to a whole number of epochs.
function experiment(preset; overrides...)
    e = from_preset(GPT2, Symbol(preset); overrides...)
    steps = e.steps ÷ epoch_steps(e) * epoch_steps(e)
    for (name, every) in (:hellaswag_every => e.hellaswag_every, :ckpt_every => e.ckpt_every)
        every % epoch_steps(e) == 0 || throw(ArgumentError(
            "$name = $every must be a multiple of the steps between validations, $(epoch_steps(e))"))
    end
    return steps == e.steps ? e : from_preset(GPT2, Symbol(preset); overrides..., steps)
end

# Trains for `steps` optimizer steps, logging to `log_dir`. The keywords `batchsize`, `steps`, `lr`,
# `eval_every`, `hellaswag_every` and `ckpt_every` override the preset's values.
function main(; preset="tinystories", data_dir=joinpath(@__DIR__, "..", "data"),
              log_dir=joinpath(@__DIR__, "log"), precision="bf16", resume=nothing, seed=0,
              peak_tflops=nothing, kws...)
    precision in PRECISIONS || throw(ArgumentError("precision must be one of $PRECISIONS"))
    overrides = (; (k => v for (k, v) in kws if v !== nothing)...)
    start_epoch = resume === nothing ? 0 : checkpoint_info(resume).epoch
    e = experiment(preset; data_dir, log_dir, seed, start_epoch, half=precision == "bf16", overrides...)
    tok = gpt2_tokenizer()

    mkpath(log_dir)
    logger = ConsoleLogger(open(joinpath(log_dir, "log.txt"), "a"), e.total_batch, 0.0, peak_tflops, time())
    Reactant.with_config(; precision_config(precision)...) do
        nitro = Nitro(e; logger, resume = resume === nothing ? false : resume)
        logger.flops = flops_per_token(nitro.model, e.cfg)
        B, T, accum = e.batchsize, e.cfg.block_size, ReactantNitro.accum(e)
        logmsg(logger, @sprintf("[setup] preset=%s params=%.1fM batch=%d×%d accum=%d tokens/step=%d precision=%s steps=%d",
                                preset, Lux.parameterlength(nitro.model) / 1e6, B, T, accum, e.total_batch,
                                precision, e.steps))
        e.eval_every > 0 && monitor_evaluations!(nitro, e, logger, tok)
        logger.t_last = time()
        train!(nitro)
    end
    close(logger.file)
    return nothing
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
            help = "number of optimizer steps, rounded down to a multiple of --eval-every (the learning-rate schedule spans them)"
            arg_type = Int
        "--lr"
            help = "peak learning rate"
            arg_type = Float32
        "--eval-every"
            help = "steps between validations, which are also epochs (0: no validation, one epoch)"
            arg_type = Int
        "--hellaswag-every"
            help = "steps between HellaSwag evaluations, a multiple of --eval-every (0 disables)"
            arg_type = Int
        "--ckpt-every"
            help = "steps between checkpoints, a multiple of --eval-every (0 disables)"
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

function (@main)(args)
    main(; parse_cli(args; description="Pretrain GPT-2 with Lux and ReactantNitro.")...)
    return nothing
end
