using Flux: BilinearUpsample
using Test

@testset "BilinearUpsample" begin
  @test size(BilinearUpsample((2, 2))(rand(2, 2, 1, 1))) == (4, 4, 1, 1)
  @test size(BilinearUpsample((3, 3))(rand(2, 2, 1, 1))) == (6, 6, 1, 1)
  @test size(BilinearUpsample((2, 2))(rand(2, 2, 10, 10))) == (4, 4, 10, 10)
  @test size(BilinearUpsample((3, 3))(rand(2, 2, 10, 10))) == (6, 6, 10, 10)

  @test_throws BoundsError BilinearUpsample((2, 2))(rand(2, 2))
end
