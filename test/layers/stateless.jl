using Flux: onehotbatch, mse, crossentropy, binarycrossentropy

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
    @test crossentropy(y_hat, y, weight = ones(2)) ≈ y_logloss
    @test crossentropy(y_hat, y, weight = [.5, .5]) ≈ y_logloss/2
    @test crossentropy(y_hat, y, weight = [2, .5]) ≈ 1.5049660054074199
  end

  @testset "binarycrossentropy" begin
    @test binarycrossentropy(y_hat, y, add=0) ≈ y_logloss
    @test mean(binarycrossentropy.(y_hat, y, add=0),2) ≈ [y_logloss; y_logloss]
    @test binarycrossentropy(0,0) ≈ -log(1+1e-7)
  end
end
