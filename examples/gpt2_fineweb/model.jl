# GPT-2: a stack of pre-LayerNorm transformer blocks, learned position embeddings, and an output
# head tied to the token embedding.

using Flux, Random, Statistics

# The two variants trained in this example. `vocab_size` pads GPT-2's 50257 tokens to a multiple of
# 64 (faster matmuls); the padding tokens never occur in the data and are excluded when sampling.
const GPT2_SMALL = (; vocab_size=50304, block_size=1024, n_layer=12, n_head=12, n_embd=768)
const GPT2_TINY  = (; vocab_size=50304, block_size=512,  n_layer=6,  n_head=6,  n_embd=384)

const NTOKENS = 50257   # GPT-2's real vocabulary

struct Block{L1, A, L2, M}
    ln1::L1
    attn::A
    ln2::L2
    mlp::M
end

Flux.@layer Block

function (b::Block)(x)
    x = x .+ b.attn(b.ln1(x))
    return x .+ b.mlp(b.ln2(x))
end

struct GPT{E, P, B, L}
    wte::E      # token embedding, also the output head
    wpe::P      # position embedding
    blocks::B
    ln_f::L
end

Flux.@layer GPT

# Init as in GPT-2: N(0, 0.02) weights, zero biases, and the projections that feed the residual
# stream scaled down by 1/√(2·n_layer), as each block adds two of them.
function GPT(cfg)
    (; vocab_size, block_size, n_layer, n_head, n_embd) = cfg
    init(dims...) = 0.02f0 .* randn32(dims...)
    resid_init(dims...) = init(dims...) ./ Float32(sqrt(2n_layer))
    blocks = map(1:n_layer) do _
        attn = Attention(n_embd; nheads=n_head, bias=true, init, is_causal=true)
        attn.out_proj.weight .= resid_init(size(attn.out_proj.weight)...)
        mlp = Chain(Dense(n_embd => 4n_embd, gelu_tanh; init),
                    Dense(4n_embd => n_embd; init=resid_init))
        Block(LayerNorm(n_embd), attn, LayerNorm(n_embd), mlp)
    end
    return GPT(Embedding(vocab_size => n_embd; init), Embedding(block_size => n_embd; init),
               Chain(blocks...), LayerNorm(n_embd))
end

# `x` holds 1-based token ids, shape (T, B). Returns the logits, shape (vocab_size, T*B).
function (m::GPT)(x::AbstractMatrix)
    T = size(x, 1)
    h = m.wte(x) .+ m.wpe.weight[:, 1:T]
    h = m.ln_f(m.blocks(h))
    return transpose(m.wte.weight) * reshape(h, size(h, 1), :)
end

# Cross-entropy of the next-token prediction, averaged over all positions, computed in Float32
# also for a bfloat16 model. Picking the target logits by linear indexing avoids a one-hot matrix,
# which Reactant would materialize at vocab_size × T·B.
function lm_loss(model, x, y)
    logits = Float32.(model(x))
    V, N = size(logits)
    target_logits = logits[vec(y) .+ V .* (0:N-1)]
    return mean(logsumexp(logits; dims=1)) - mean(target_logits)
end

# Sample `n` continuations of `prompt` (0-based ids) with top-k sampling. Runs eagerly, so pass a
# CPU copy of the model. There is no KV cache: each new token re-runs the whole context, which is
# cheap for the short samples logged during training.
function generate(model, prompt::AbstractVector{<:Integer}; n=4, len=32, topk=50,
                  rng=Random.default_rng())
    block_size = size(model.wpe.weight, 2)
    x = repeat(prompt .+ 1, 1, n)
    for _ in 1:len
        ctx = x[max(1, end - block_size + 1):end, :]
        logits = reshape(model(ctx), :, size(ctx)...)[1:NTOKENS, end, :]
        next = [sample_topk(rng, logits[:, j], topk) for j in 1:n]
        x = vcat(x, permutedims(next))
    end
    return [x[:, j] .- 1 for j in 1:n]
end

function sample_topk(rng, logits, k)
    idx = partialsortperm(logits, 1:k; rev=true)
    p = softmax(logits[idx])
    return idx[something(findfirst(≥(rand(rng)), cumsum(p)), k)]
end
