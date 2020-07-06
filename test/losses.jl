using Test
using Flux: onehotbatch, σ

using Flux.Losses: mse, crossentropy, logitcrossentropy, binarycrossentropy, logitbinarycrossentropy
using Flux.Losses: xlogx, xlogy

# group here all losses, used in tests
const ALL_LOSSES = [Flux.Losses.mse, Flux.Losses.mae, Flux.Losses.msle,
                    Flux.Losses.crossentropy, Flux.Losses.logitcrossentropy,
                    Flux.Losses.binarycrossentropy, Flux.Losses.logitbinarycrossentropy, 
                    Flux.Losses.kldivergence,
                    Flux.Losses.huber_loss,
                    Flux.Losses.tversky_loss,
                    Flux.Losses.dice_coeff_loss,
                    Flux.Losses.poisson_loss,
                    Flux.Losses.hinge_loss, Flux.Losses.squared_hinge_loss]


@testset "xlogx & xlogy" begin
  @test iszero(xlogx(0))
  @test isnan(xlogx(NaN))
  @test xlogx(2) ≈ 2.0 * log(2.0)
  @inferred xlogx(2)
  @inferred xlogx(0)

  @test iszero(xlogy(0, 1))
  @test isnan(xlogy(NaN, 1))
  @test isnan(xlogy(1, NaN))
  @test isnan(xlogy(NaN, NaN))
  @test xlogy(2, 3) ≈ 2.0 * log(3.0)
  @inferred xlogy(2, 3)
  @inferred xlogy(0, 1)
end

# First, regression-style y's
y = [1, 1, 0, 0]
ŷ = [.9, .1, .1, .9]

@testset "mse" begin
  @test mse(ŷ, y) ≈ (.1^2 + .9^2)/2
end

@testset "mae" begin
  @test Flux.mae(ŷ, y) ≈ 1/2
end

@testset "huber_loss" begin
  @test Flux.huber_loss(ŷ, y) ≈ 0.20500000000000002
end

y = [123.0,456.0,789.0]
ŷ = [345.0,332.0,789.0]
@testset "msle" begin
  @test Flux.msle(ŷ, y) ≈ 0.38813985859136585
end

# Now onehot y's
y = onehotbatch([1, 1, 0, 0], 0:1)
ŷ = [.1 .9; .9 .1; .9 .1; .1 .9]'
v = log(.1 / .9)
logŷ = [v 0.0; 0.0 v; 0.0 v; v 0.0]'
lossvalue = 1.203972804325936

@testset "crossentropy" begin
  @test crossentropy([0.1,0.0,0.9], [0.1,0.0,0.9]) ≈ crossentropy([0.1,0.9], [0.1,0.9])
  @test crossentropy(ŷ, y) ≈ lossvalue
end

@testset "logitcrossentropy" begin
  @test logitcrossentropy(logŷ, y) ≈ lossvalue
end

logŷ, y = randn(3), rand(3)

@testset "binarycrossentropy" begin
  @test binarycrossentropy(σ.(logŷ), y; ϵ=0) ≈ mean(-y.*log.(σ.(logŷ)) - (1 .- y).*log.(1 .- σ.(logŷ)))
  @test binarycrossentropy(σ.(logŷ), y) ≈ mean(-y.*log.(σ.(logŷ) .+ eps.(σ.(logŷ))) - (1 .- y).*log.(1 .- σ.(logŷ) .+ eps.(σ.(logŷ))))
end

@testset "logitbinarycrossentropy" begin
  @test logitbinarycrossentropy(logŷ, y) ≈ binarycrossentropy(σ.(logŷ), y; ϵ=0)
end

y = [1 2 3]
ŷ = [4.0 5.0 6.0]

@testset "kldivergence" begin
  @test Flux.kldivergence([0.1,0.0,0.9], [0.1,0.0,0.9]) ≈ Flux.kldivergence([0.1,0.9], [0.1,0.9])
  @test Flux.kldivergence(ŷ, y) ≈ -1.7661057888493457
  @test Flux.kldivergence(y, y) ≈ 0
end

y = [1 2 3 4]
ŷ = [5.0 6.0 7.0 8.0]

@testset "hinge_loss" begin
  @test Flux.hinge_loss(ŷ, y) ≈ 0
  @test Flux.hinge_loss(y, 0.5 .* y) ≈ 0.125
end

@testset "squared_hinge_loss" begin
  @test Flux.squared_hinge_loss(ŷ, y) ≈ 0
  @test Flux.squared_hinge_loss(y, 0.5 .* y) ≈ 0.0625
end

y = [0.1 0.2 0.3]
ŷ = [0.4 0.5 0.6]

@testset "poisson_loss" begin
  @test Flux.poisson_loss(ŷ, y) ≈ 0.6278353988097339
  @test Flux.poisson_loss(y, y) ≈ 0.5044459776946685
end

y = [1.0 0.5 0.3 2.4]
ŷ = [0 1.4 0.5 1.2]

@testset "dice_coeff_loss" begin
  @test Flux.dice_coeff_loss(ŷ, y) ≈ 0.2799999999999999
  @test Flux.dice_coeff_loss(y, y) ≈ 0.0
end

@testset "tversky_loss" begin
  @test Flux.tversky_loss(ŷ, y) ≈ -0.06772009029345383
  @test Flux.tversky_loss(ŷ, y, β=0.8) ≈ -0.09490740740740744
  @test Flux.tversky_loss(y, y) ≈ -0.5576923076923075
end

@testset "no spurious promotions" begin
  for T in (Float32, Float64)
    y = rand(T, 2)
    ŷ = rand(T, 2)
    for f in ALL_LOSSES
      fwd, back = Flux.pullback(f, ŷ, y)
      @test fwd isa T
      @test eltype(back(one(T))[1]) == T
    end
  end
end
