# HellaSwag (https://rowanzellers.com/hellaswag): pick the most likely of 4 endings for a context.
# Each ending is scored by the model's mean per-token loss on it, given the context ("acc_norm"
# in build-nanogpt, where GPT-2 124M reaches 0.2955).

if abspath(PROGRAM_FILE) == @__FILE__
    get!(ENV, "TF_CPP_MIN_LOG_LEVEL", "1")
    using PythonCall   # before Flux, see train.jl
    include("model.jl")
    include("data.jl")
end

using Flux, HuggingFaceDatasets, Reactant, Statistics

# Tokenized HellaSwag rows, 4 per example, right-padded to a common length `Tmax`: `x` and `y` are
# the inputs and next-token targets, and `mask` selects the target positions inside the ending.
function hellaswag_data(tok; split="validation", group=16, pad_multiple=32)
    ds = load_dataset("Rowan/hellaswag"; split)
    ctxs, endings = collect(ds["ctx"]), collect(ds["endings"])
    labels = parse.(Int, collect(ds["label"])) .+ 1
    rows = Vector{Int}[]
    masks = Vector{Bool}[]
    for (ctx, ends) in zip(ctxs, endings)
        c = tokenize(tok, ctx)
        for e in ends
            ids = tokenize(tok, " " * e)
            push!(rows, vcat(c, ids))
            push!(masks, vcat(falses(length(c)), trues(length(ids))))
        end
    end
    n = length(labels)
    nex = cld(n, group) * group      # pad the example count to whole groups
    Tmax = cld(maximum(length, rows), pad_multiple) * pad_multiple
    tokens = fill(EOT, Tmax, 4nex)
    mask = zeros(Float32, Tmax, 4nex)
    for (j, (r, m)) in enumerate(zip(rows, masks))
        tokens[1:length(r), j] .= r
        mask[1:length(m), j] .= m
    end
    return (; x=tokens[1:end-1, :], y=tokens[2:end, :], mask=mask[2:end, :], labels, n, group)
end

# The summed next-token loss of each row over its masked positions.
function ending_losses(model, x, y, mask)
    logits = Float32.(model(x))
    V, N = size(logits)
    nll = vec(logsumexp(logits; dims=1)) .- logits[vec(y) .+ V .* (0:N-1)]
    return vec(sum(reshape(nll, size(x)) .* mask; dims=1))
end

# A function `model -> (; acc, acc_norm)` giving the HellaSwag accuracy (`acc` sums the ending
# loss, `acc_norm` averages it), scoring batches of `data.group` examples moved to `device`. On
# Reactant the scoring is compiled once, for models with the structure of `model`.
function hellaswag_evaluator(model, data; device=identity)
    (; x, y, mask, labels, n, group) = data
    cols(b) = (4group*(b-1)+1):(4group*b)
    batch(b) = device((x[:, cols(b)], y[:, cols(b)], mask[:, cols(b)]))
    score = if Flux.get_device_type(model) <: Flux.ReactantDevice
        Reactant.@compile ending_losses(model, batch(1)...)
    else
        ending_losses
    end
    counts = reshape(vec(sum(mask; dims=1))[1:4n], 4, n)
    return function (model)
        losses = reduce(vcat, (Array(score(model, batch(b)...)) for b in 1:cld(n, group)))
        sums = reshape(losses[1:4n], 4, n)
        acc = mean(argmin.(eachcol(sums)) .== labels)
        acc_norm = mean(argmin.(eachcol(sums ./ counts)) .== labels)
        return (; acc, acc_norm)
    end
end

# `julia --project=. hellaswag.jl <checkpoint.jld2>`
if abspath(PROGRAM_FILE) == @__FILE__
    using JLD2
    ckpt = JLD2.load(ARGS[1])
    dev = reactant_device()
    model = Flux.loadmodel!(GPT(ckpt["cfg"]), ckpt["model_state"]) |> dev
    (; acc, acc_norm) = hellaswag_evaluator(model, hellaswag_data(gpt2_tokenizer()); device=dev)(model)
    println("HellaSwag (step $(ckpt["step"])): acc = $(round(acc, digits=4)), acc_norm = $(round(acc_norm, digits=4))")
end
