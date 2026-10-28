# HellaSwag (https://rowanzellers.com/hellaswag): pick the most likely of 4 endings for a context.
# Each ending is scored by the model's mean per-token loss on it, given the context ("acc_norm"
# in build-nanogpt, where GPT-2 124M reaches 0.2955). The tokenized examples are made by
# `hellaswag_data` in ../flux-reactant/data.jl.

if abspath(PROGRAM_FILE) == @__FILE__
    get!(ENV, "TF_CPP_MIN_LOG_LEVEL", "1")
    include("experiment.jl")
end

using Statistics

# The summed next-token loss of each row over its masked positions.
function ending_losses(half, model, ps, st, x, y, mask)
    logits, _ = Lux.apply(model, x, half ? bf16_params(ps) : ps, st)
    return vec(sum(token_nll(logits, y) .* mask; dims=1))
end

# A function `ps -> (; acc, acc_norm)` giving the HellaSwag accuracy (`acc` sums the ending loss,
# `acc_norm` averages it), scoring batches of `data.group` examples with the model of parameters
# `ps` and state `st`, on the device. The scoring function is compiled once, for parameters like `ps`.
function hellaswag_evaluator(model, ps, st, data; half=true)
    (; x, y, mask, labels, n, group) = data
    cols(b) = (4group*(b-1)+1):(4group*b)
    batch(b) = Reactant.to_rarray((x[:, cols(b)], y[:, cols(b)], mask[:, cols(b)]))
    score = Reactant.compile((ps, st, xb, yb, mb) -> ending_losses(half, model, ps, st, xb, yb, mb),
                             (ps, st, batch(1)...))
    counts = reshape(vec(sum(mask; dims=1))[1:4n], 4, n)
    return function (ps)
        losses = reduce(vcat, (Array(score(ps, st, batch(b)...)) for b in 1:cld(n, group)))
        sums = reshape(losses[1:4n], 4, n)
        acc = mean(argmin.(eachcol(sums)) .== labels)
        acc_norm = mean(argmin.(eachcol(sums ./ counts)) .== labels)
        return (; acc, acc_norm)
    end
end

# The configuration of the model with parameters `ps`, whose heads are 64-dimensional, as in both
# presets: a checkpoint holds the parameters, not the configuration.
function gpt_config(ps)
    n_embd, vocab_size = size(ps.wte.weight)
    return (; vocab_size, block_size=size(ps.wpe.weight, 2), n_layer=length(ps.blocks),
              n_head=n_embd ÷ 64, n_embd)
end

# Prints the HellaSwag accuracy of the model saved in the checkpoint at `path`, in Float32.
function hellaswag_checkpoint(path)
    rec = load_checkpoint(Checkpointer(), path)
    model = GPT(gpt_config(rec.ps))
    ps, st = Reactant.to_rarray((rec.ps, rec.st))
    evaluate = hellaswag_evaluator(model, ps, st, hellaswag_data(gpt2_tokenizer()); half=false)
    (; acc, acc_norm) = evaluate(ps)
    println("HellaSwag (step $(rec.step)): acc = $(round(acc, digits=4)), acc_norm = $(round(acc_norm, digits=4))")
end

# `julia --project=. hellaswag.jl <checkpoint>`
if abspath(PROGRAM_FILE) == @__FILE__
    hellaswag_checkpoint(ARGS[1])
end
