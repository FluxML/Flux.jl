using Test
using Flux: flatten

@testset "helpers" begin
    @testset "flatten" begin
        x = randn(Float32, 10, 10, 3, 2)
        @test size(flatten(x)) == (300, 2)
    end

    @testset "normalise" begin
        x = randn(Float32, 3, 2, 2)
        @test Flux.normalise(x) == Flux.normalise(x; dims = 3)
    end
end
