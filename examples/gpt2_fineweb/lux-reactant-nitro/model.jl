# GPT-2 in Lux: a stack of pre-LayerNorm transformer blocks, learned position embeddings, and an
# output head tied to the token embedding. The architecture and the initialization are those of
# ../flux-reactant/model.jl.

using Lux, Random
using NNlib: dot_product_attention, gelu_tanh, logsumexp
using BFloat16s: BFloat16
using MLUtils: sample_logits

const GPT2_SMALL = (; vocab_size=50304, block_size=1024, n_layer=12, n_head=12, n_embd=768)
const GPT2_TINY  = (; vocab_size=50304, block_size=512,  n_layer=6,  n_head=6,  n_embd=384)

const NTOKENS = 50257   # GPT-2's real vocabulary

# Causal multi-head self-attention, with the queries, keys and values of all heads in one projection.
struct CausalAttention{Q, P} <: Lux.AbstractLuxContainerLayer{(:qkv, :proj)}
    nheads::Int
    qkv::Q
    proj::P
end

function (m::CausalAttention)(x, ps, st)
    qkv, st_qkv = m.qkv(x, ps.qkv, st.qkv)
    n = size(qkv, 1) ÷ 3
    T = size(x, 2)
    mask = reshape(1:T, :, 1) .<= reshape(1:T, 1, :)   # mask[j, i]: position i attends to j <= i
    y, _ = dot_product_attention(qkv[1:n, :, :], qkv[n+1:2n, :, :], qkv[2n+1:3n, :, :];
                                 nheads=m.nheads, mask)
    y, st_proj = m.proj(y, ps.proj, st.proj)
    return y, (; qkv=st_qkv, proj=st_proj)
end

struct Block{L1, A, L2, M} <: Lux.AbstractLuxContainerLayer{(:ln1, :attn, :ln2, :mlp)}
    ln1::L1
    attn::A
    ln2::L2
    mlp::M
end

# LayerNorm runs in Float32, whatever the eltype of the residual stream `x`, and its output goes back
# to that eltype.
function layernorm(ln, x, ps, st)
    h, st = ln(Float32.(x), ps, st)
    return eltype(x).(h), st
end

function (b::Block)(x, ps, st)
    h, st_ln1 = layernorm(b.ln1, x, ps.ln1, st.ln1)
    a, st_attn = b.attn(h, ps.attn, st.attn)
    x = x .+ a
    h, st_ln2 = layernorm(b.ln2, x, ps.ln2, st.ln2)
    m, st_mlp = b.mlp(h, ps.mlp, st.mlp)
    return x .+ m, (; ln1=st_ln1, attn=st_attn, ln2=st_ln2, mlp=st_mlp)
end

struct GPT{E, P, B, L} <: Lux.AbstractLuxContainerLayer{(:wte, :wpe, :blocks, :ln_f)}
    wte::E      # token embedding, also the output head
    wpe::P      # position embedding
    blocks::B
    ln_f::L
end

# Init as in GPT-2: N(0, 0.02) weights, zero biases, and the projections that feed the residual
# stream scaled down by 1/√(2·n_layer), as each block adds two of them.
function GPT(cfg)
    (; vocab_size, block_size, n_layer, n_head, n_embd) = cfg
    init(rng, dims...) = 0.02f0 .* randn32(rng, dims...)
    resid_init(rng, dims...) = init(rng, dims...) ./ Float32(sqrt(2n_layer))
    blocks = map(1:n_layer) do _
        attn = CausalAttention(n_head, Dense(n_embd => 3n_embd; init_weight=init),
                               Dense(n_embd => n_embd; init_weight=resid_init))
        mlp = Chain(Dense(n_embd => 4n_embd, gelu_tanh; init_weight=init),
                    Dense(4n_embd => n_embd; init_weight=resid_init))
        Block(LayerNorm((n_embd, 1); dims=1), attn, LayerNorm((n_embd, 1); dims=1), mlp)
    end
    return GPT(Embedding(vocab_size => n_embd; init_weight=init),
               Embedding(block_size => n_embd; init_weight=init),
               Chain(blocks...), LayerNorm((n_embd, 1); dims=1))
end

# `x` holds 1-based token ids, shape (T, B). Returns the logits, shape (vocab_size, T, B).
function (m::GPT)(x::AbstractMatrix, ps, st)
    T = size(x, 1)
    tok, st_wte = m.wte(x, ps.wte, st.wte)
    h = tok .+ ps.wpe.weight[:, 1:T]
    h, st_blocks = m.blocks(h, ps.blocks, st.blocks)
    h, st_ln_f = layernorm(m.ln_f, h, ps.ln_f, st.ln_f)
    logits = reshape(transpose(ps.wte.weight) * reshape(h, size(h, 1), :), :, size(h)[2:end]...)
    return logits, (; wte=st_wte, wpe=st.wpe, blocks=st_blocks, ln_f=st_ln_f)
end

# The parameters in bfloat16, but for the LayerNorm ones, which are the only ones with a `scale`.
# The optimizer keeps and updates the Float32 parameters: the model sees a bfloat16 copy.
bf16_params(x::AbstractArray) = BFloat16.(x)
bf16_params(ps::NamedTuple) = haskey(ps, :scale) ? ps : map(bf16_params, ps)

# The cross-entropy of the next-token prediction at each position, in Float32 also for bfloat16
# `logits`, shape (vocab_size, size(y)...): `logsumexp(logits) - logits[y]`.
function token_nll(logits, y)
    V = size(logits, 1)
    idx = vec(y) .+ V .* (0:length(y)-1)
    lse = vec(logsumexp(Float32.(logits); dims=1))
    picked = Float32.(vec(logits)[idx])   # not fused into the subtraction: Reactant fails to trace it
    return reshape(lse .- picked, size(y))
end

# The parameters of the model that go into matrix multiplications (everything but the position
# embedding, which is a lookup), for the FLOPs of a training step per token: 6 per parameter
# (forward + backward) plus the attention scores.
function flops_per_token(model, cfg)
    n = Lux.parameterlength(model) - cfg.block_size * cfg.n_embd
    return 6n + 12 * cfg.n_layer * cfg.n_embd * cfg.block_size
end

# Sample `n` continuations of `prompt` with top-k sampling. Runs eagerly, on the CPU. There is no KV
# cache: each new token re-runs the whole context, which is cheap for the short samples logged
# during training.
function generate(model, ps, st, prompt::AbstractVector{<:Integer}; n=4, len=32, topk=50,
                  rng=Random.default_rng())
    block_size = size(ps.wpe.weight, 2)
    x = repeat(prompt, 1, n)
    for _ in 1:len
        ctx = x[max(1, end - block_size + 1):end, :]
        logits = first(model(ctx, ps, st))[1:NTOKENS, end, :]
        x = vcat(x, sample_logits(rng, logits; k=topk))
    end
    return [x[:, j] for j in 1:n]
end
