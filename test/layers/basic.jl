using Test
using Flux: activations


@testset "basic" begin
    @testset "Chain" begin
        @test_nowarn Chain(Dense(10, 5, σ),Dense(5, 2), softmax)
        @test_nowarn Chain(Dense(10, 5, σ),Dense(5, 2), softmax)(randn(10))
    end

    @testset "Dense" begin
        @test  length(Dense(10, 5)(randn(10))) == 5
        @test_throws DimensionMismatch Dense(10, 5)(randn(1))
        Random.seed!(0)
        @test all(Dense(10, 1)(randn(10)).data .≈ 1.1774348382231168)
        Random.seed!(0)
        @test all(Dense(10, 2)(randn(10)).data .≈ [  -0.3624741476779616
            -0.46724765394534323])

        @test_throws DimensionMismatch Dense(10, 5)(1)
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

    @testset "activations" begin
        c = Chain(Dense(10, 5, σ),Dense(5, 2), softmax)
        # Single layer activation
        @test length(activations(c[1], randn(10))) == 1
        @test isa(activations(c[1], randn(10)), Array{Any,1})
        # Chain activation
        @test length(activations(c, randn(10))) == 3
        @test isa(activations(c, randn(10)), Array{Any,1})
    end
end
