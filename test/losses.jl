using Test
using Flux: onehotbatch, σ
using Statistics: mean

using Flux.Losses: mse, label_smoothing, crossentropy, logitcrossentropy, binarycrossentropy, logitbinarycrossentropy
using Flux.Losses: xlogx, xlogy

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

  # Test that mse() loss works on complex values:
  @test mse(0 + 0im, 1 + 1im) == 2
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
y_smoothed = label_smoothing(y, 0.1)
ŷ = [.1 .9; .9 .1; .9 .1; .1 .9]'
v = log(.1 / .9)
logŷ = [v 0.0; 0.0 v; 0.0 v; v 0.0]'
lossvalue = 1.203972804325936
lossvalue_smoothed = 1.2039728043259348
yl = onehotbatch([1], 0:1)
sf = 0.1
yls = [sf (1-sf)]'  # Effective y after label smoothing
ylp = [0.9 0.1]'
logylp = [0.0 v]'

# Construct `sim`ilar and `dis`imilar versions of the dataset so we can test effect of smoothing
# smoothing should decrease loss on disimilar and increase the loss on similar, compared to
# the loss without smoothing
ya = onehotbatch([1, 1, 1, 0, 0], 0:1)
ya_smoothed = label_smoothing(ya, 2sf)
y_same = Float32.(ya)
y_sim = y_same .* (1-2*sf) .+ sf
y_dis = copy(y_sim)
y_dis[1,:], y_dis[2,:] = y_dis[2,:], y_dis[1,:]

@testset "crossentropy" begin
  @test crossentropy([0.1,0.0,0.9], [0.1,0.0,0.9]) ≈ crossentropy([0.1,0.9], [0.1,0.9])
  @test crossentropy(ŷ, y) ≈ lossvalue
  @test crossentropy(ŷ, y_smoothed) ≈ lossvalue_smoothed
  @test crossentropy(ylp, label_smoothing(yl, 2sf)) ≈ -sum(yls.*log.(ylp))
  @test crossentropy(ylp, yl) ≈ -sum(yl.*log.(ylp))
  @test iszero(crossentropy(y_same, ya, eps=0))
  @test iszero(crossentropy(ya, ya, eps=0))
  @test crossentropy(y_sim, ya) < crossentropy(y_sim, ya_smoothed)
  @test crossentropy(y_dis, ya) > crossentropy(y_dis, ya_smoothed)
end

@testset "logitcrossentropy" begin
  @test logitcrossentropy(logŷ, y) ≈ lossvalue
  @test logitcrossentropy(logylp, yl) ≈ -sum(yl.*logsoftmax(logylp))
  @test logitcrossentropy(logylp, label_smoothing(yl, 2sf)) ≈ -sum(yls.*logsoftmax(logylp))
end

logŷ, y = randn(3), rand(3)
yls = y.*(1-2sf).+sf

@testset "binarycrossentropy" begin
  @test binarycrossentropy.(σ.(logŷ), label_smoothing(y, 2sf; dims=0); eps=0) ≈ -yls.*log.(σ.(logŷ)) - (1 .- yls).*log.(1 .- σ.(logŷ))
  @test binarycrossentropy(σ.(logŷ), y; eps=0) ≈ mean(-y.*log.(σ.(logŷ)) - (1 .- y).*log.(1 .- σ.(logŷ)))
  @test binarycrossentropy(σ.(logŷ), y) ≈ mean(-y.*log.(σ.(logŷ) .+ eps.(σ.(logŷ))) - (1 .- y).*log.(1 .- σ.(logŷ) .+ eps.(σ.(logŷ))))
  @test binarycrossentropy([0.1,0.2,0.9], 1) ≈ -mean(log, [0.1,0.2,0.9])  # constant label
end

@testset "logitbinarycrossentropy" begin
  @test logitbinarycrossentropy.(logŷ, label_smoothing(y, 0.2)) ≈ binarycrossentropy.(σ.(logŷ), label_smoothing(y, 0.2); eps=0)
  @test logitbinarycrossentropy(logŷ, y) ≈ binarycrossentropy(σ.(logŷ), y; eps=0)
end

y = onehotbatch([1], 0:1)
yls = [0.1 0.9]'
@testset "label_smoothing" begin
  @test label_smoothing(y, 0.2) == yls
  @test label_smoothing(y, 0.2; dims=0) == label_smoothing.(y, 0.2; dims=0)
  @test_throws ArgumentError label_smoothing([0., 0., 1., 0.], 1.2)
  @test_throws ArgumentError label_smoothing([0., 0., 1., 0.], 0.)
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
  @test Flux.tversky_loss(ŷ, y, beta=0.8) ≈ -0.09490740740740744
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

@testset "binary_focal_loss" begin
    y = [0  1  0
         1  0  1]
    ŷ = [0.268941  0.5  0.268941
         0.731059  0.5  0.731059]

    y1 = [1 0
          0 1]
    ŷ1 = [0.6 0.3
          0.4 0.7]
    @test Flux.binary_focal_loss(ŷ, y) ≈ 0.0728675615927385
    @test Flux.binary_focal_loss(ŷ1, y1) ≈ 0.05691642237852222
    @test Flux.binary_focal_loss(ŷ, y; gamma=0.0) ≈ Flux.binarycrossentropy(ŷ, y)
end

@testset "focal_loss" begin
    y = [1  0  0  0  1
         0  1  0  1  0
         0  0  1  0  0]
    ŷ = softmax(reshape(-7:7, 3, 5) .* 1f0)
    y1 = [1 0
          0 0
          0 1]
    ŷ1 = [0.4 0.2
          0.5 0.5
          0.1 0.3]
    @test Flux.focal_loss(ŷ, y) ≈ 1.1277571935622628
    @test Flux.focal_loss(ŷ1, y1) ≈ 0.45990566879720157
    @test Flux.focal_loss(ŷ, y; gamma=0) ≈ Flux.crossentropy(ŷ, y)
end
  
@testset "siamese_contrastive_loss" begin
  y = [1 0
       0 0
       0 1]
  ŷ = [0.4 0.2
       0.5 0.5
       0.1 0.3]
  y1 = [1 0 0 0 1
        0 1 0 1 0
        0 0 1 0 0]
  ŷ1 = softmax(reshape(-7:7, 3, 5) .* 1.0f0)
  y2 = [1
        0
        0
        1
        1]
  ŷ2 = [0.6
        0.4
        0.1
        0.2
        0.7]
  @test Flux.siamese_contrastive_loss(ŷ, y) ≈ 0.2333333333333333
  @test Flux.siamese_contrastive_loss(ŷ, y, margin = 0.5f0) ≈ 0.10000000000000002
  @test Flux.siamese_contrastive_loss(ŷ, y, margin = 1.5f0) ≈ 0.5333333333333333
  @test Flux.siamese_contrastive_loss(ŷ1, y1) ≈ 0.32554644f0
  @test Flux.siamese_contrastive_loss(ŷ1, y1, margin = 0.5f0) ≈ 0.16271012f0
  @test Flux.siamese_contrastive_loss(ŷ1, y1, margin = 1.5f0) ≈ 0.6532292f0
  @test Flux.siamese_contrastive_loss(ŷ, y, margin = 1) ≈ Flux.siamese_contrastive_loss(ŷ, y)
  @test Flux.siamese_contrastive_loss(y, y) ≈ 0.0
  @test Flux.siamese_contrastive_loss(y1, y1) ≈ 0.0
  @test Flux.siamese_contrastive_loss(ŷ, y, margin = 0) ≈ 0.09166666666666667
  @test Flux.siamese_contrastive_loss(ŷ1, y1, margin = 0) ≈ 0.13161165f0
  @test Flux.siamese_contrastive_loss(ŷ2, y2) ≈ 0.21200000000000005
  @test Flux.siamese_contrastive_loss(ŷ2, ŷ2) ≈ 0.18800000000000003
  @test_throws DomainError(-0.5, "Margin must be non-negative") Flux.siamese_contrastive_loss(ŷ1, y1, margin = -0.5)
  @test_throws DomainError(-1, "Margin must be non-negative") Flux.siamese_contrastive_loss(ŷ, y, margin = -1)
end
