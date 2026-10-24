@testset "mooncake gradient" begin
    for (model, x, name) in TEST_MODELS
        @testset "grad check $name" begin
            @test test_gradients(model, x; reference=AutoZygote(), compare=AutoMooncake())
        end
    end
end

@testset "mooncake withgradient auxiliary output" begin
    # scalar output (no aux)
    g0 = Flux.withgradient(x -> sum(2 .* x), AutoMooncake(), [1.0, 2.0, 3.0])
    @test g0.val ≈ 12.0
    @test g0.grad[1] ≈ [2.0, 2.0, 2.0]

    # Tuple aux output: the first element is the loss, the rest are auxiliary.
    g1 = Flux.withgradient(AutoMooncake(), [1.0, 2.0, 4.0]) do x
        z = 1 ./ x
        sum(z), z  # here z is an auxillary output
    end
    @test g1.grad[1] ≈ [-1.0, -0.25, -0.0625]
    @test g1.val[1] ≈ 1.75
    @test g1.val[2] ≈ [1.0, 0.5, 0.25]

    # NamedTuple aux output, including a non-differentiable auxiliary.
    g2 = Flux.withgradient(AutoMooncake(), [1.0, 2.0, 4.0]) do x
        z = 1 ./ x
        (loss=sum(z), aux=string(z))
    end
    @test g2.grad[1] ≈ [-1.0, -0.25, -0.0625]
    @test g2.val.loss ≈ 1.75
    @test g2.val.aux == "[1.0, 0.5, 0.25]"
end

@testset "mooncake gradient through f64" begin
    W = randn(Float32, 4, 4)
    m = Chain(Dense(W, true, tanh), LayerNorm(4), Dense(W))  # tied weight
    x = randn(Float32, 4, 3)
    loss(m) = sum(abs2, m(x))
    g = Flux.gradient(loss, AutoMooncake(), m)[1]
    g64 = Flux.gradient(m -> loss(f64(m)), AutoMooncake(), m)[1]
    @test eltype(g64.layers[1].weight) == Float32
    @test g64.layers[1].weight ≈ g.layers[1].weight
    @test g64.layers[2].diag.scale ≈ g.layers[2].diag.scale
end
