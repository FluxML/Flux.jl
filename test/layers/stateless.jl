using Flux: onehotbatch, mse, crossentropy, weighted_crossentropy

@testset "losses" begin
  # First, regression-style y's
  y = [1, 1, 0, 0]
  y_hat = [.9, .1, .1, .9]

  @testset "mse" begin
    @test mse(y_hat, y) ≈ (.1^2 + .9^2)/2
  end

  # Now onehot y's
  y = onehotbatch([1, 1, 0, 0], 0:1)
  y_hat = [.1 .9; .9 .1; .9 .1; .1 .9]'
  y_logloss = 1.203972804325936

  @testset "crossentropy" begin
    @test crossentropy(y_hat, y) ≈ y_logloss
  end

  @testset "weighted_crossentropy" begin
    @test weighted_crossentropy(y_hat, y, ones(2)) ≈ y_logloss
    @test weighted_crossentropy(y_hat, y, [.5, .5]) ≈ y_logloss/2
    @test weighted_crossentropy(y_hat, y, [2, .5]) ≈ 1.5049660054074199
  end
end
