using Flux, Test
using Flux: maxpool, meanpool

@testset "Pooling" begin
  x = randn(10, 10, 3, 2)
  mp = MaxPool((2, 2))
  @test mp(x) == maxpool(x, (2,2))
  mp = MeanPool((2, 2))
  @test mp(x) == meanpool(x, (2,2))
end

@testset "CNN" begin
  r = zeros(28, 28, 1, 5)
  m = Chain(
    Conv((2, 2), 1=>16, relu),
    MaxPool((2,2)),
    Conv((2, 2), 16=>8, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10), softmax)

  @test size(m(r)) == (10, 5)
end
