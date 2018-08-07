using Test
using Flux
using Flux: ctc
using Requires

@testset "ctc" begin

  x = [1. 2. 3.; 2. 1. 1.; 3. 3. 2.]
  y = [1 0 0; 1 0 0; 0 1 0]
  lossvalue = 3.6990738275138035
  gradvalues = [-0.317671 -0.427729 0.665241; 0.244728 -0.0196172 -0.829811; 0.0729422 0.447346 0.16457]

  l, gs = ctc(x, y)
  @test l ≈ lossvalue
  @test any(x -> !x, gs ≈ gradvalues)
  @require CUDAnative begin
    @require CuArrays begin
      lossvalue = 3.6990738
      l, gs = ctc(Flux.gpu(x), Flux.gpu(y))
      @test l ≈ lossvalue
      @test gs ≈ gradvalues 
    end
  end
end
