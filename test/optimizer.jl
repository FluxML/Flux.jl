@testset "training julia models" begin

    @testset "linear regression" begin
        srand(0)

        model = Affine(10, 1)

        truth = Float32[0, 4, 2, 2, -3, 6, -1, 3, 2, -5]'

        data = map(1:256) do i
            x = rand(Float32, 10)
            x, truth * x + 3rand(Float32)
        end

        Flux.train!(model, data, epoch=5)

        @test cor(reshape.((model.W.x, truth), 10)...) > .99
    end

    @testset "logistic regression" begin
        srand(0)

        model = Chain(Affine(10, 1), σ)

        truth = Float32[0, 4, 2, 2, -3, 6, -1, 3, 2, -5]'

        data = map(1:256) do i
            x = rand(Float32, 10)
            x, truth * x + 2rand(Float32) > 5f0
        end

        Flux.train!(model, data, epoch=10)

        @test cor(reshape.((model.layers[1].W.x, truth), 10)...) > .99
    end

end

