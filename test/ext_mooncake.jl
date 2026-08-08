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
