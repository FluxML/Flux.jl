using Test
using Flux: flatten

@testset "helpers" begin
  @testset "flatten" begin
    x = randn(Float32, 10, 10, 3, 2)
    @test size(flatten(x)) == (300, 2)
  end
end
