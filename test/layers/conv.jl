using Test
using Flux: Chain, Conv, MaxPool, MeanPool
using Base.conv

@testset "pooling" begin
  mp = MaxPool((2, 2))

  @testset "maxpooling" begin
    @test MaxPool{2}(2) == mp
    @test MaxPool{2}(2; pad=1, stride=3) == MaxPool((2, 2); pad=(1, 1), stride=(3, 3))
  end

  mp = MeanPool((2, 2))

  @testset "meanpooling" begin
    @test MeanPool{2}(2) == mp
    @test MeanPool{2}(2; pad=1, stride=3) == MeanPool((2, 2); pad=(1, 1), stride=(3, 3))
  end
end

@testset "cnn" begin
    r = zeros(28, 28)
    m = Chain(
      Conv((2, 2), 1=>16, relu),
      MaxPool{2}(2),
      Conv((2, 2), 16=>8, relu),
      MaxPool{2}(2),
      x -> reshape(x, :, size(x, 4)),
      Dense(288, 10), softmax)

  @testset "inference" begin
    @test size(m(r)) == (10, )
  end
end
