using Test
using Flux: flatten

@testset "helpers" begin
  @testset "flatten" begin
    x = randn(Float32, 10, 10, 3, 2)
    @test size(flatten(x)) == (300, 2)
  end

  @testset "normalize" begin
    x = randn(Float32, 3, 2, 2)
    @test normalize(x) == normalize(x; dims=3)
  end
end
