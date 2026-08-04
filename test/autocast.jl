# This testset MUST run before any `autocast` call in this file: it checks that until
# the first use flips `Flux.autocast_active()`, the machinery is compiled out entirely
# and layers infer their exact concrete return types.
@testset "autocast is compiled out before first use" begin
    @test Flux.autocast_active() == false
    @test @inferred(Dense(3 => 4, relu)(randn(Float32, 3, 8))) isa Matrix{Float32}
    @test @inferred(Conv((3,), 2 => 4, relu)(randn(Float32, 10, 2, 5))) isa Array{Float32, 3}
    @test @inferred(Chain(Dense(3 => 4, relu), Dense(4 => 2))(randn(Float32, 3, 8))) isa Matrix{Float32}
end

@testset "autocast eltype flow ($T)" for T in (Float16, BFloat16)
    x2 = randn(Float32, 3, 8)                 # for Dense-like layers
    x4 = randn(Float32, 8, 8, 2, 3)           # for conv layers
    xseq = randn(Float32, 3, 7, 2)            # for recurrent layers

    @testset "cast-down layers" begin
        for (l, x) in (
            (Dense(3 => 4, relu), x2),
            (Flux.Bilinear((3, 3) => 4), x2),
            (Conv((3, 3), 2 => 4, relu), x4),
            (ConvTranspose((3, 3), 2 => 4), x4),
            (CrossCor((3, 3), 2 => 4), x4),
            (RNN(3 => 5), xseq),
            (LSTM(3 => 5), xseq),
            (GRU(3 => 5), xseq),
            (GRUv3(3 => 5), xseq),
        )
            y = autocast(() -> l(x), T)
            @test eltype(y) == T
            # parameters are untouched
            @test all(p -> eltype(p) == Float32, Flux.trainables(l))
        end

        mha = MultiHeadAttention(16)
        xmha = randn(Float32, 16, 5, 2)
        y, α = autocast(() -> mha(xmha), T)
        @test eltype(y) == T

        e = Embedding(5 => 4)
        @test eltype(autocast(() -> e(Flux.onehotbatch([1, 3], 1:5)), T)) == T
        @test eltype(autocast(() -> e([1, 3]), T)) == Float32  # gather path stays Float32
    end

    @testset "normalization computes in Float32" begin
        for (l, x) in (
            (BatchNorm(3), x2),
            (LayerNorm(3), x2),
            (InstanceNorm(2; affine=true), x4),
            (GroupNorm(2, 2), x4),
        )
            y = autocast(() -> l(x), T)
            @test eltype(y) == Float32
        end
        # half-precision input to a norm layer is upcast inside the scope
        xh = T == Float16 ? f16(x2) : bf16(x2)
        @test eltype(autocast(() -> LayerNorm(3)(xh), T)) == Float32
    end

    @testset "losses upcast to Float32" begin
        half = T == Float16 ? f16 : bf16
        ŷ, y = half(rand(Float32, 4, 8)), half(rand(Float32, 4, 8))
        for loss in (Flux.mse, Flux.mae, Flux.crossentropy, Flux.logitcrossentropy,
                     Flux.huber_loss)
            @test autocast(() -> loss(ŷ, y), T) isa Float32
            @test loss(ŷ, y) isa T  # unchanged outside the scope
        end
    end

    @testset "gradients are Float32 and close to the fp32 reference" begin
        model = Chain(Dense(3 => 4, relu), BatchNorm(4), Dense(4 => 2))
        ytarget = randn(Float32, 2, 8)
        loss(m) = Flux.mse(m(x2), ytarget)

        val, grad = Flux.withgradient(loss, model; autocast=T)
        @test val isa Float32
        gflat = filter(g -> g isa AbstractArray, Functors.fleaves(grad[1]))
        @test !isempty(gflat)
        @test all(g -> eltype(g) == Float32, gflat)
        @test all(g -> all(isfinite, g), gflat)

        val32, grad32 = Flux.withgradient(loss, model)
        rtol = T == Float16 ? 0.03 : 0.15
        @test val ≈ val32 rtol=rtol
        @test grad[1].layers[1].weight ≈ grad32[1].layers[1].weight rtol=rtol atol=0.05

        # raw reductions (not Flux losses) stay in half precision, like PyTorch
        vraw, _ = Flux.withgradient(m -> sum(abs2, m(x2)), model; autocast=T)
        @test vraw isa T
    end

    @testset "do-block and keyword forms agree" begin
        model = Chain(Dense(3 => 4, tanh), Dense(4 => 2))
        loss(m) = Flux.mse(m(x2), zeros(Float32, 2, 8))
        g1 = autocast(() -> Flux.gradient(loss, model), T)
        g2 = Flux.gradient(loss, model; autocast=T)
        @test g1[1].layers[1].weight == g2[1].layers[1].weight
        wg = Flux.withgradient(loss, AutoZygote(), model; autocast=T)
        @test wg.grad[1].layers[1].weight == g2[1].layers[1].weight
    end

    @testset "train! with autocast" begin
        model = Chain(Dense(3 => 4, relu), Dense(4 => 2))
        w0 = copy(model[1].weight)
        opt = Flux.setup(Adam(1e-3), model)
        data = [(randn(Float32, 3, 8), randn(Float32, 2, 8)) for _ in 1:3]
        Flux.train!((m, x, y) -> Flux.mse(m(x), y), model, data, opt; autocast=T)
        @test eltype(model[1].weight) == Float32
        @test model[1].weight != w0
    end
end

@testset "autocast is a no-op outside the scope" begin
    model = Chain(Dense(3 => 4, relu), BatchNorm(4), Dense(4 => 2))
    x = randn(Float32, 3, 8)
    y0 = model(x)
    autocast(() -> model(x), Float16)  # entering and leaving a scope changes nothing
    @test model(x) == y0
    @test Flux.mse(y0, zero(y0)) isa Float32
end

@testset "forward pass infers as a small union after first use" begin
    # The earlier testsets flipped `autocast_active()`. From then on the inferred return
    # is at worst the 3-type union over the Float32/Float16/BFloat16 paths (which the
    # compiler union-splits) — not `Any` — and the union must not widen through a Chain.
    @test Flux.autocast_active() == true
    MatUnion3 = Union{Matrix{Float32}, Matrix{Float16}, Matrix{BFloat16}}
    @test Base.promote_op(Dense(3 => 4, relu), Matrix{Float32}) <: MatUnion3
    @test Base.promote_op(Conv((3,), 2 => 4, relu), Array{Float32, 3}) <:
        Union{Array{Float32, 3}, Array{Float16, 3}, Array{BFloat16, 3}}
    @test Base.promote_op(Chain(Dense(3 => 4, relu), Dense(4 => 2)), Matrix{Float32}) <: MatUnion3
end

@testset "autocast argument checking" begin
    @test_throws ArgumentError autocast(() -> 1, Float32)
    @test_throws ArgumentError autocast(() -> 1, Float64)
    @test_throws ArgumentError autocast(() -> 1, Int)
end

@testset "outputsize under autocast" begin
    model = Chain(Dense(3 => 4), Conv((3, 3), 1 => 2))
    m2 = Chain(Dense(3 => 7), Dense(7 => 2))
    @test autocast(() -> outputsize(m2, (3, 5)), Float16) == (2, 5)
end

@testset "autocast with Mooncake" begin
    model = Chain(Dense(3 => 4, tanh), Dense(4 => 2))
    x = randn(Float32, 3, 8)
    loss(m) = Flux.mse(m(x), zeros(Float32, 2, 8))
    g16 = Flux.gradient(loss, AutoMooncake(config=nothing), model; autocast=Float16)
    g32 = Flux.gradient(loss, model)
    @test eltype(g16[1].layers[1].weight) == Float32
    @test g16[1].layers[1].weight ≈ g32[1].layers[1].weight rtol=0.03 atol=0.05
end
