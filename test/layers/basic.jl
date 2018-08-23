using Test, Random


@testset "basic" begin
    @testset "Chain" begin
        @test_nowarn Chain(Dense(10, 5, σ),Dense(5, 2))(randn(10))
        @test_throws DimensionMismatch Chain(Dense(10, 5, σ),Dense(2, 1))(randn(10))
        # numeric test should be put into testset of corresponding layer
    end

    @testset "Dense" begin
        @test  length(Dense(10, 5)(randn(10))) == 5
        @test_throws DimensionMismatch Dense(10, 5)(randn(1))
        @test_throws DimensionMismatch Dense(10, 5)(1) # avoid broadcasting
        @test_throws DimensionMismatch Dense(10, 5).(randn(10)) # avoid broadcasting

        Random.seed!(0)
        @test all(Dense(10, 1)(randn(10)).data .≈ 1.1774348382231168)
        Random.seed!(0)
        @test all(Dense(10, 2)(randn(10)).data .≈ [  -0.3624741476779616
            -0.46724765394534323])

    end

    @testset "Diagonal" begin
        @test length(Flux.Diagonal(10)(randn(10))) == 10
        @test length(Flux.Diagonal(10)(1)) == 10
        @test length(Flux.Diagonal(10)(randn(1))) == 10
        @test_throws DimensionMismatch Flux.Diagonal(10)(randn(2))
        Random.seed!(0)
        @test all(Flux.Diagonal(2)(randn(2)).data .≈ [ 0.6791074260357777,
            0.8284134829000359])
    end
end
