@testset "inference stays tight" begin
    # Plain layers are unaffected — exact concrete return type.
    @test @inferred(Dense(3 => 4, relu)(randn(Float32, 3, 8))) isa Matrix{Float32}
    @test @inferred(Conv((3,), 2 => 4, relu)(randn(Float32, 10, 2, 5))) isa Array{Float32, 3}
    @test @inferred(Chain(Dense(3 => 4, relu), Dense(4 => 2))(randn(Float32, 3, 8))) isa Matrix{Float32}

    # Wrapped layers also infer their exact half-precision return type (the whole point of
    # the wrapper design over a runtime scope).
    for T in (Float16, BFloat16)
        @test @inferred(autocast(Dense(3 => 4, relu), T)(randn(Float32, 3, 8))) isa Matrix{T}
        @test @inferred(autocast(Conv((3,), 2 => 4, relu), T)(randn(Float32, 10, 2, 5))) isa Array{T, 3}
        # a wrapped norm layer computes in Float32
        xh = T == Float16 ? f16(randn(Float32, 4, 8)) : bf16(randn(Float32, 4, 8))
        @test @inferred(autocast(LayerNorm(4), T)(xh)) isa Matrix{Float32}
    end
end

@testset "eltype flow ($T)" for T in (Float16, BFloat16)
    x2 = randn(Float32, 3, 8)
    x4 = randn(Float32, 8, 8, 2, 3)
    xseq = randn(Float32, 3, 7, 2)

    @testset "down-cast layers → $T" begin
        for (l, x) in ((Dense(3 => 4, relu), x2),
                       (Flux.Bilinear((3, 3) => 4), x2),
                       (Conv((3, 3), 2 => 4, relu), x4),
                       (ConvTranspose((3, 3), 2 => 4), x4),
                       (CrossCor((3, 3), 2 => 4), x4),
                       (RNN(3 => 5), xseq),
                       (LSTM(3 => 5), xseq),
                       (GRU(3 => 5), xseq),
                       (GRUv3(3 => 5), xseq))
            @test eltype(autocast(l, T)(x)) == T
            @test all(p -> eltype(p) == Float32, Flux.trainables(l))  # params untouched
        end
        # MultiHeadAttention: its projections are wrapped, so attention runs in T
        mha = MultiHeadAttention(16)
        y, α = autocast(mha, T)(randn(Float32, 16, 5, 2))
        @test eltype(y) == T
    end

    @testset "norm layers → Float32" begin
        for (l, x) in ((BatchNorm(3), x2),
                       (LayerNorm(3), x2),
                       (InstanceNorm(2; affine=true), x4),
                       (GroupNorm(2, 2), x4))
            @test eltype(autocast(l, T)(x)) == Float32
        end
    end

    @testset "Embedding is not wrapped (kept full precision)" begin
        e = Embedding(5 => 4)
        @test Flux.autocast_mode(e) == :none
        me = autocast(e, T)
        @test eltype(me(Flux.onehotbatch([1, 3], 1:5))) == Float32
        @test eltype(me([1, 3])) == Float32
    end
end

@testset "losses always accumulate in Float32" begin
    for cast in (f16, bf16)
        ŷ, y = cast(rand(Float32, 4, 8)), cast(rand(Float32, 4, 8))
        for loss in (Flux.mse, Flux.mae, Flux.crossentropy, Flux.logitcrossentropy, Flux.huber_loss)
            @test loss(ŷ, y) isa Float32
        end
    end
    # Float32 inputs are unaffected
    @test Flux.mse(rand(Float32, 4), rand(Float32, 4)) isa Float32
end

@testset "gradients via the autocast keyword ($T)" for T in (Float16, BFloat16)
    x = randn(Float32, 3, 8)
    ytarget = randn(Float32, 2, 8)
    model = Chain(Dense(3 => 4, relu), BatchNorm(4), Dense(4 => 2))
    loss(m) = Flux.mse(m(x), ytarget)

    val, grad = Flux.withgradient(loss, model; autocast=T)
    @test val isa Float32
    # grad tree matches the *unwrapped* model, so the usual opt_state works
    g32 = Flux.gradient(loss, model)[1]
    @test typeof(grad[1].layers[1]) == typeof(g32.layers[1])
    gflat = filter(g -> g isa AbstractArray, Functors.fleaves(grad[1]))
    @test !isempty(gflat)
    @test all(g -> eltype(g) == Float32, gflat)
    @test all(g -> all(isfinite, g), gflat)
    rtol = T == Float16 ? 0.03 : 0.15
    @test grad[1].layers[1].weight ≈ g32.layers[1].weight rtol=rtol atol=0.05

    # keyword form ≡ explicitly wrapping the model. `Chain` itself is not wrapped (its
    # layers are), so the grad of the wrapped model nests as layers → AutocastDown → layer.
    g_explicit = Flux.gradient(m -> loss(m), autocast(model, T))[1]
    @test g_explicit.layers[1].layer.weight ≈ grad[1].layers[1].weight rtol=1e-5
end

@testset "train! with autocast" begin
    model = Chain(Dense(3 => 4, relu), BatchNorm(4), Dense(4 => 2))
    w0 = copy(model[1].weight)
    opt = Flux.setup(Adam(1e-3), model)
    data = [(randn(Float32, 3, 8), randn(Float32, 2, 8)) for _ in 1:3]
    Flux.train!((m, x, y) -> Flux.mse(m(x), y), model, data, opt; autocast=BFloat16)
    @test eltype(model[1].weight) == Float32
    @test model[1].weight != w0
end

@testset "argument checking" begin
    @test_throws ArgumentError autocast(Dense(2 => 2), Float32)
    @test_throws ArgumentError autocast(Dense(2 => 2), Float64)
    @test_throws ArgumentError autocast(Dense(2 => 2), Int)
end

@testset "outputsize through a wrapped model" begin
    m = autocast(Chain(Dense(3 => 7), Dense(7 => 2)), Float16)
    @test outputsize(m, (3, 5)) == (2, 5)
end

@testset "custom layer opts in via autocast_mode" begin
    struct MyLinear{W}; weight::W; end
    Flux.@layer MyLinear
    (l::MyLinear)(x) = l.weight * x
    Flux.autocast_mode(::MyLinear) = :down

    l = MyLinear(randn(Float32, 4, 3))
    x = randn(Float32, 3, 8)
    @test eltype(autocast(l, Float16)(x)) == Float16
    @test eltype(l.weight) == Float32
    g = Flux.gradient(m -> sum(abs2, m(x)), l; autocast=Float16)[1]
    @test eltype(g.weight) == Float32
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
