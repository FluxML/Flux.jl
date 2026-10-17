attn_weights(m, args...; kws...) = last(attention_with_weights(m, args...; kws...))

# the output is the projection of the values averaged with the weights
function output_from_weights(m, v_in, α)
  v = reshape(m.v_proj(v_in), :, m.nheads, size(v_in)[2:end]...)
  x = permutedims(batched_mul(permutedims(v, (1, 3, 2, 4)), α), (1, 3, 2, 4))
  return m.out_proj(reshape(x, :, size(x)[3:end]...))
end

@testset "Attention" begin
  dim = 4; nheads = 2; len = 3; batch_size = 5
  attn = Attention(dim; nheads)
  q = rand(Float32, (dim, len, batch_size))
  k = rand(Float32, (dim, len + 1, batch_size))
  v = rand(Float32, (dim, len + 1, batch_size))

  y = attn(q, k, v)
  @test y isa Array{Float32, 3}
  @test size(y) == (dim, len, batch_size)

  @testset "self-attention" begin
    @test attn(q) ≈ attn(q, q, q)
  end

  @testset "key and value are the same" begin
    @test attn(q, k) ≈ attn(q, k, k)
  end

  @testset "change dims" begin
    attn2 = Attention(4 => 10 => 5; nheads)
    @test size(attn2(q, k, v)) == (5, len, batch_size)

    attn3 = Attention((4, 3, 2) => (6, 8) => 5; nheads)
    k3 = rand(Float32, (3, len + 1, batch_size))
    v3 = rand(Float32, (2, len + 1, batch_size))
    @test size(attn3(q, k3, v3)) == (5, len, batch_size)

    @test_throws ArgumentError Attention(5; nheads)
    @test_throws ArgumentError Attention(4; nheads = 4, nkvheads = 3)
  end

  @testset "attention_with_weights" begin
    y2, α = attention_with_weights(attn, q, k, v)
    @test y2 ≈ y
    @test α isa Array{Float32, 4}
    @test size(α) == (len + 1, len, nheads, batch_size)
    @test all(≈(1), sum(α; dims = 1))
    @test output_from_weights(attn, v, α) ≈ y
    @test attention_with_weights(attn, q)[1] ≈ attn(q)
    @test attention_with_weights(attn, q, k)[1] ≈ attn(q, k)
    @test attn_weights(attn, q, k) ≈ α
  end

  @testset "is_causal" begin
    causal = Attention(dim; nheads, is_causal = true)
    Flux.loadmodel!(causal, attn)
    mask = NNlib.make_causal_mask(q; dims = 2)
    @test causal(q) ≈ attn(q; mask)
    @test causal(q) ≈ attn(q; is_causal = true)
    @test causal(q; is_causal = false) ≈ attn(q)

    α = attn_weights(causal, q)
    @test all(α[2, 1, :, :] .== 0)
    @test α[:, :, 1, 1] ≈ triu(α[:, :, 1, 1])

    # a position does not affect the preceding ones
    q2 = copy(q)
    q2[:, end, :] .= 0
    @test causal(q2)[:, 1:end-1, :] ≈ causal(q)[:, 1:end-1, :]
    @test !(attn(q2)[:, 1:end-1, :] ≈ attn(q)[:, 1:end-1, :])

    # together with a mask
    keep = trues(len, batch_size)
    keep[end, 1] = false
    pad = reshape(keep, len, 1, 1, batch_size)
    @test causal(q; mask = pad) ≈ attn(q; mask = pad .& mask)
    @test attn_weights(causal, q; mask = pad) ≈ attn_weights(attn, q; mask = pad .& mask)
  end

  @testset "mask" begin
    # exclude the last key of each sequence
    keep = trues(len + 1, batch_size)
    keep[end, :] .= false
    mask = reshape(keep, len + 1, 1, 1, batch_size)
    α = attn_weights(attn, q, k; mask)
    @test all(α[end, :, :, :] .== 0)

    k2, v2 = copy(k), copy(v)
    k2[:, end, :] .= 0
    v2[:, end, :] .= 0
    @test attn(q, k2, v2; mask) ≈ attn(q, k, v; mask)
    @test !(attn(q, k2, v2) ≈ attn(q, k, v))
  end

  @testset "sequence lengths" begin
    kv_lengths = [len + 1, len, 2, len + 1, 1]
    mask = reshape((1:len + 1) .<= kv_lengths', len + 1, 1, 1, batch_size)
    @test attn(q, k, v; kv_lengths) ≈ attn(q, k, v; mask)
    @test attn_weights(attn, q, k; kv_lengths) ≈ attn_weights(attn, q, k; mask)
    @test attn(q, k, v; kv_lengths = len) ≈ attn(q, k[:, 1:len, :], v[:, 1:len, :])

    q_lengths = [len, 2, 1, len, len]
    y_pad = attn(q, k, v; q_lengths)
    @test all(y_pad[:, 3, 2] .== 0)
    @test all(y_pad[:, 2:3, 3] .== 0)
    @test y_pad[:, 1:2, 2] ≈ attn(q, k, v)[:, 1:2, 2]
    α = attn_weights(attn, q, k; q_lengths)
    @test all(α[:, 3, :, 2] .== 0)

    causal = Attention(dim; nheads, is_causal = true)
    lengths = [len, 2, 1, len, len]
    causal_mask = NNlib.make_causal_mask(q; dims = 2)
    pad = reshape((1:len) .<= lengths', len, 1, 1, batch_size)
    @test causal(q; kv_lengths = lengths) ≈ causal(q; mask = pad .& causal_mask)

    test_gradients(attn, q, k, v; loss = (m, q, k, v) -> mean(m(q, k, v; kv_lengths)))
  end

  @testset "bias" begin
    # use bias to produce a causal mask
    b = fill(-Inf32, (len, len))
    for i in 1:len, j in i:len
      b[i, j] = 0
    end
    @test attn(q; bias = b) ≈ attn(q; is_causal = true)
    @test attn_weights(attn, q; bias = b) ≈ attn_weights(attn, q; is_causal = true)
  end

  @testset "grouped-query attention" begin
    gqa = Attention(8; nheads = 4, nkvheads = 2)
    @test size(gqa.q_proj.weight) == (8, 8)
    @test size(gqa.k_proj.weight) == (4, 8)
    @test size(gqa.v_proj.weight) == (4, 8)
    x = rand(Float32, (8, len, batch_size))
    @test size(gqa(x)) == (8, len, batch_size)
    @test size(attn_weights(gqa, x)) == (len, len, 4, batch_size)

    # same as full attention where each key/value head is repeated for its group of query heads
    full = Attention(8; nheads = 4)
    repeat_heads(w) = reshape(repeat(reshape(w, 2, 2, 8); inner = (1, 2, 1)), 8, 8)
    full.q_proj.weight .= gqa.q_proj.weight
    full.k_proj.weight .= repeat_heads(gqa.k_proj.weight)
    full.v_proj.weight .= repeat_heads(gqa.v_proj.weight)
    full.out_proj.weight .= gqa.out_proj.weight
    @test full(x) ≈ gqa(x)
    @test attn_weights(full, x) ≈ attn_weights(gqa, x)
  end

  @testset "qk_norm" begin
    normed = Attention(8; nheads = 2, qk_norm = LayerNorm)
    @test normed.q_norm isa LayerNorm
    @test normed.k_norm isa LayerNorm
    @test normed.q_norm !== normed.k_norm
    @test size(normed.q_norm.diag.scale) == (4,)
    @test attn.q_norm === identity

    x = rand(Float32, (8, len, batch_size))
    @test size(normed(x)) == (8, len, batch_size)

    # the weights do not depend on the scale of the queries
    α = attn_weights(normed, x)
    normed.q_proj.weight .*= 3
    @test attn_weights(normed, x) ≈ α rtol=1e-3
    test_gradients(normed, x)
  end

  @testset "batch dimensions" begin
    q2 = rand(Float32, (dim, len))
    k2 = rand(Float32, (dim, len + 1))
    @test attn(q2, k2) ≈ dropdims(attn(reshape(q2, dim, len, 1), reshape(k2, dim, len + 1, 1)); dims = 3)
    @test size(attn_weights(attn, q2, k2)) == (len + 1, len, nheads)

    q4 = rand(Float32, (dim, len, 2, 3))
    @test attn(q4) ≈ reshape(attn(reshape(q4, dim, len, 6)), dim, len, 2, 3)
    @test size(attn_weights(attn, q4)) == (len, len, nheads, 2, 3)
  end

  @testset "dropout" begin
    drop = Attention(dim; nheads, dropout = 0.5)
    @test drop(q) ≈ drop(q)  # inactive outside of a gradient call
    Flux.trainmode!(drop)
    @test !(drop(q) ≈ drop(q))
    Flux.testmode!(drop)
    @test drop(q) ≈ drop(q)

    # in automatic mode it is active within a gradient call
    Flux.testmode!(drop, :auto)
    l, g = withgradient(m -> sum(abs2, m(q)), drop)
    @test !(l ≈ sum(abs2, drop(q)))
    @test g[1].q_proj.weight isa Matrix{Float32}

    # the weights are returned after the dropout
    Flux.trainmode!(drop)
    y_drop, α_drop = attention_with_weights(drop, q)
    @test any(iszero, α_drop)
    @test output_from_weights(drop, q, α_drop) ≈ y_drop
  end

  @testset "show" begin
    @test repr(Attention(8; nheads = 2)) == "Attention(8; nheads=2)"
    @test repr(Attention(8 => 16 => 4; nheads = 2)) == "Attention(8 => 16 => 4; nheads=2)"
    @test repr(Attention((3, 4, 5) => (6, 8) => 7; nheads = 2, nkvheads = 1, bias = true,
                         dropout = 0.1, qk_norm = LayerNorm, is_causal = true)) ==
          "Attention((3, 4, 5) => (6, 8) => 7; nheads=2, nkvheads=1, bias=true, dropout=0.1, qk_norm=LayerNorm, is_causal=true)"
  end

  @testset "outputsize" begin
    @test outputsize(attn, (dim, len, batch_size)) == (dim, len, batch_size)
    m = @autosize (4, 3, 5) Attention(_ => 8 => 6; nheads = 2)
    @test size(m.q_proj.weight) == (8, 4)
    @test size(m(q)) == (6, len, batch_size)
  end

  @testset "gradient" begin
    test_gradients(attn, q)
    test_gradients(attn, q, k, v)
    test_gradients(Attention(dim; nheads, is_causal = true), q)
    test_gradients(Attention(8; nheads = 4, nkvheads = 2), rand(Float32, (8, len, batch_size)))
  end
end

@testset "MultiHeadAttention (deprecated)" begin
  dim = 4; nheads = 2; len = 3; batch_size = 5
  mha = @test_deprecated MultiHeadAttention(dim; nheads)
  q = rand(Float32, (dim, len, batch_size))
  k = rand(Float32, (dim, len, batch_size))
  v = rand(Float32, (dim, len, batch_size))

  y, α = mha(q, k, v)
  @test y isa Array{Float32, 3}
  @test size(y) == (dim, len, batch_size)
  @test α isa Array{Float32, 4}
  @test size(α) == (len, len, nheads, batch_size)

  @testset "self-attention" begin
    y1, α1 = mha(q)
    y2, α2 = mha(q, q, q)
    @test y1 ≈ y2
    @test α1 ≈ α2
  end

  @testset "key and value are the same" begin
    y1, α1 = mha(q, k)
    y2, α2 = mha(q, k, k)
    @test y1 ≈ y2
    @test α1 ≈ α2
  end

  @testset "change dims" begin
    dims = 4 => 10 => 5
    mha2 = @test_deprecated MultiHeadAttention(dims; nheads)
    y2, _ = mha2(q, k, v)
    @test size(y2) == (dims.second.second, len, batch_size)
  end

  @testset "mask" begin
    mask = NNlib.make_causal_mask(q; dims = 2)
    y, α = mha(q; mask)
    @test all(α[2, 1, :, :] .== 0)
    @test α[:, :, 1, 1] ≈ triu(α[:, :, 1, 1])
  end

  @testset "bias" begin
    # use bias to produce a causal mask
    b = zeros(Float32, (len, len))
    for i in 1:len, j in i:len
      b[i, j] = typemax(Float32)
    end
    y, α = mha(q, k, v, b)
    @test all(α[2, 1, :, :] .== 0)
    @test α[:, :, 1, 1] ≈ triu(α[:, :, 1, 1])
  end

  @testset "gradient" begin
    function loss(m, q)
      # `local`, otherwise the outer `y`, `α` are captured and Enzyme returns zero gradients
      local y, α = m(q)
      return sum(y.^2) + sum(α.^2)
    end
    test_gradients(mha, q; loss)
  end

  @testset "conversion to Attention" begin
    y0, α0 = mha(q, k, v)
    attn = Attention(mha)
    @test attn.q_proj === mha.q_proj
    @test attn(q, k, v) ≈ y0
    y1, α1 = attention_with_weights(attn, q, k, v)
    @test y1 ≈ y0
    @test α1 ≈ α0

    attn2 = Attention(dim; nheads)
    Flux.loadmodel!(attn2, mha)
    @test attn2(q, k, v) ≈ y0

    attn3 = Attention(dim; nheads)
    Flux.loadmodel!(attn3, Flux.state(mha))
    @test attn3(q, k, v) ≈ y0
  end
end
