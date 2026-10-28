# HellaSwag (https://rowanzellers.com/hellaswag): pick the most likely of 4 endings for a context.
# Each ending is scored by the model's mean per-token loss on it, given the context ("acc_norm"
# in build-nanogpt, where GPT-2 124M reaches 0.2955).

if abspath(PROGRAM_FILE) == @__FILE__
    get!(ENV, "TF_CPP_MIN_LOG_LEVEL", "1")
    using PythonCall   # before Flux, see train.jl
    include("model.jl")
    include("data.jl")
    using Reactant
end

using Flux, JLD2, Statistics

# The summed next-token loss of each row over its masked positions.
function ending_losses(model, x, y, mask)
    nll = Flux.logitcrossentropy(model(x), vec(y); agg=identity)
    return vec(sum(reshape(nll, size(x)) .* mask; dims=1))
end

# A function `model -> (; acc, acc_norm)` giving the HellaSwag accuracy (`acc` sums the ending
# loss, `acc_norm` averages it), scoring batches of `data.group` examples moved to `device`.
# The scoring function is `compile(ending_losses, args)` for arguments `args` like those it gets:
# `Reactant.compile` compiles it once, for models with the structure of `model`.
function hellaswag_evaluator(model, data; device=identity, compile=(f, args) -> f)
    (; x, y, mask, labels, n, group) = data
    cols(b) = (4group*(b-1)+1):(4group*b)
    batch(b) = device((x[:, cols(b)], y[:, cols(b)], mask[:, cols(b)]))
    score = compile(ending_losses, (model, batch(1)...))
    counts = reshape(vec(sum(mask; dims=1))[1:4n], 4, n)
    return function (model)
        losses = reduce(vcat, (Array(score(model, batch(b)...)) for b in 1:cld(n, group)))
        sums = reshape(losses[1:4n], 4, n)
        acc = mean(argmin.(eachcol(sums)) .== labels)
        acc_norm = mean(argmin.(eachcol(sums ./ counts)) .== labels)
        return (; acc, acc_norm)
    end
end

# Prints the HellaSwag accuracy of the model saved in the checkpoint at `path`, in Float32.
function hellaswag_checkpoint(path; device, compile=(f, args) -> f)
    ckpt = JLD2.load(path)
    model = Flux.loadmodel!(GPT(ckpt["cfg"]), ckpt["model_state"]) |> device
    evaluate = hellaswag_evaluator(model, hellaswag_data(gpt2_tokenizer()); device, compile)
    (; acc, acc_norm) = evaluate(model)
    println("HellaSwag (step $(ckpt["step"])): acc = $(round(acc, digits=4)), acc_norm = $(round(acc_norm, digits=4))")
end

# `julia --project=. hellaswag.jl <checkpoint.jld2>`
if abspath(PROGRAM_FILE) == @__FILE__
    hellaswag_checkpoint(ARGS[1]; device=reactant_device(), compile=Reactant.compile)
end
