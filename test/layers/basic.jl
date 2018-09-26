using Test, Random

@testset "basic" begin
    @testset "Chain" begin
        @test_nowarn Chain(Dense(10, 5, σ), Dense(5, 2))(randn(10))
        @test_throws DimensionMismatch Chain(Dense(10, 5, σ),Dense(2, 1))(randn(10))
        # numeric test should be put into testset of corresponding layer
    end

    @testset "Dense" begin
        @test  length(Dense(10, 5)(randn(10))) == 5
        @test_throws DimensionMismatch Dense(10, 5)(randn(1))
        @test_throws MethodError Dense(10, 5)(1) # avoid broadcasting
        @test_throws MethodError Dense(10, 5).(randn(10)) # avoid broadcasting

        @test Dense(10, 1, identity, initW = ones, initb = zeros)(ones(10,1)) == [10]
        @test Dense(10, 1, identity, initW = ones, initb = zeros)(ones(10,2)) == [10 10]
        @test Dense(10, 2, identity, initW = ones, initb = zeros)(ones(10,1)) == [10; 10]
        @test Dense(10, 2, identity, initW = ones, initb = zeros)([ones(10,1) 2*ones(10,1)]) == [10 20; 10 20]

    end

    @testset "Diagonal" begin
        @test length(Flux.Diagonal(10)(randn(10))) == 10
        @test length(Flux.Diagonal(10)(1)) == 10
        @test length(Flux.Diagonal(10)(randn(1))) == 10
        @test_throws DimensionMismatch Flux.Diagonal(10)(randn(2))

        @test Flux.Diagonal(2)([1 2]) == [1 2; 1 2]
        @test Flux.Diagonal(2)([1,2]) == [1,2]
        @test Flux.Diagonal(2)([1 2; 3 4]) == [1 2; 3 4]
    end
end
