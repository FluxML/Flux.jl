using Base.Test
using Flux: onehotbatch, mse, crossentropy, logitcrossentropy, 
            σ, binarycrossentropy, logitbinarycrossentropy

@testset "losses" begin
  # First, regression-style y's
  y = [1, 1, 0, 0]
  ŷ = [.9, .1, .1, .9]

  @testset "mse" begin
    @test mse(ŷ, y) ≈ (.1^2 + .9^2)/2
  end

  # Now onehot y's
  y = onehotbatch([1, 1, 0, 0], 0:1)
  ŷ = [.1 .9; .9 .1; .9 .1; .1 .9]'
  v = log(.1 / .9)
  logŷ = [v 0.0; 0.0 v; 0.0 v; v 0.0]'
  lossvalue = 1.203972804325936

  @testset "crossentropy" begin
    @test crossentropy(ŷ, y) ≈ lossvalue
  end

  @testset "logitcrossentropy" begin
    @test logitcrossentropy(logŷ, y) ≈ lossvalue
  end

  @testset "weighted_crossentropy" begin
    @test crossentropy(ŷ, y, weight = ones(2)) ≈ lossvalue
    @test crossentropy(ŷ, y, weight = [.5, .5]) ≈ lossvalue/2
    @test crossentropy(ŷ, y, weight = [2, .5]) ≈ 1.5049660054074199
  end

  @testset "weighted_logitcrossentropy" begin
    @test logitcrossentropy(logŷ, y, weight = ones(2)) ≈ lossvalue
    @test logitcrossentropy(logŷ, y, weight = [.5, .5]) ≈ lossvalue/2
    @test logitcrossentropy(logŷ, y, weight = [2, .5]) ≈ 1.5049660054074199
  end

  @testset "binarycrossentropy" begin
    @test binarycrossentropy(ŷ, y, meps = 0) ≈ lossvalue
    @test mean(binarycrossentropy.(ŷ, y, meps = 0),2) ≈ [lossvalue; lossvalue]
    @test binarycrossentropy(0.0, 0.0) ≈ -log(eps(eltype(0.0)) + 1)
    logŷ, y = randn(3), rand(3)
    @test binarycrossentropy.(σ.(logŷ), y, meps = 0) ≈ -y.*log.(σ.(logŷ)) - (1 - y).*log.(1 - σ.(logŷ))
  end
  
  @testset "logitbinarycrossentropy" begin
    @test logitbinarycrossentropy.(logŷ, y) ≈ binarycrossentropy.(σ.(logŷ), y, meps = 0)
  end

end
