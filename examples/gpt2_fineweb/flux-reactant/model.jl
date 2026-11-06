# GPT-2: a stack of pre-LayerNorm transformer blocks, learned position embeddings, and an output
# head tied to the token embedding.

using Flux, Random
using NNlib: scaled_dot_product_attention
using MLUtils: chunk, sample_logits

# The two variants trained in this example. `vocab_size` pads GPT-2's 50257 tokens to a multiple of
# 64 (faster matmuls); the padding tokens never occur in the data and are excluded when sampling.
const GPT2_SMALL = (; vocab_size=50304, block_size=1024, n_layer=12, n_head=12, n_embd=768)
const GPT2_TINY  = (; vocab_size=50304, block_size=512,  n_layer=6,  n_head=6,  n_embd=384)

const NTOKENS = 50257   # GPT-2's real vocabulary

# Causal multi-head self-attention, with the queries, keys and values of all heads in one projection.
struct CausalAttention{Q, P}
    nheads::Int
    qkv::Q
    proj::P
end

Flux.@layer CausalAttention

# Splitting with `chunk` rather than indexing is for Zygote: its pullback concatenates the three
# gradients, where indexing would scatter each into a zero array. The copies are dense, as cuDNN's
# attention requires. Reactant compiles both to the same program.
function (m::CausalAttention)(x)
    heads(a) = reshape(copy(a), :, m.nheads, size(a, 2), size(a, 3))   # (head_dim, nheads, T, B)
    q, k, v = map(heads, chunk(m.qkv(x), 3; dims=1))
    y = scaled_dot_product_attention(q, k, v; is_causal=true)
    return m.proj(reshape(y, :, size(y, 3), size(y, 4)))
end

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
        attn = CausalAttention(n_head, Dense(n_embd => 3n_embd; init),
                               Dense(n_embd => n_embd; init=resid_init))
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

# Cross-entropy of the next-token prediction, averaged over all positions (in Float32 also for a
# bfloat16 model).
lm_loss(model, x, y) = Flux.logitcrossentropy(model(x), vec(y))

# Sample `n` continuations of `prompt` with top-k sampling. Runs eagerly, so pass a CPU copy of
# the model. There is no KV cache: each new token re-runs the whole context, which is cheap for the
# short samples logged during training.
function generate(model, prompt::AbstractVector{<:Integer}; n=4, len=32, topk=50,
                  rng=Random.default_rng())
    block_size = size(model.wpe.weight, 2)
    x = repeat(prompt, 1, n)
    for _ in 1:len
        ctx = x[max(1, end - block_size + 1):end, :]
        logits = reshape(model(ctx), :, size(ctx)...)[1:NTOKENS, end, :]
        x = vcat(x, sample_logits(rng, logits; k=topk))
    end
    return [x[:, j] for j in 1:n]
end
