using Test
using Flux: onehotbatch, mse, crossentropy, logitcrossentropy,
            σ, binarycrossentropy, logitbinarycrossentropy

const ϵ = 1e-7

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
  yl = onehotbatch([1], 0:1)
  ls = 0.1
  yls = [ls (1-ls)]'  # Effective y after label smoothing
  ylp = [0.9 0.1]'
  logylp = [0.0 v]'

  # Construct `sim`ilar and `dis`imilar versions of the dataset so we can test effect of smoothing
  # smoothing should decrease loss on disimilar and increase the loss on similar, compared to 
  # the loss without smoothing
  ya = onehotbatch([1, 1, 1, 0, 0], 0:1)
  y_same = Float32.(ya)
  y_sim = y_same .* (1-2*ls) .+ ls
  y_dis = copy(y_sim)
  y_dis[1,:], y_dis[2,:] = y_dis[2,:], y_dis[1,:]

  @testset "crossentropy" begin
    @test crossentropy(ŷ, y) ≈ lossvalue
    @test crossentropy(ylp, yl, label_smoothing=2*ls) ≈ -sum(yls.*log.(ylp))
    @test crossentropy(ylp, yl) ≈ -sum(yl.*log.(ylp))
    @test iszero(crossentropy(y_same, ya))
    @test iszero(crossentropy(ya, ya))
    @test crossentropy(y_sim, ya) < crossentropy(y_sim, ya, label_smoothing=2*ls)
    @test crossentropy(y_dis, ya) > crossentropy(y_dis, ya, label_smoothing=2*ls)
  end

  @testset "logitcrossentropy" begin
    @test logitcrossentropy(logŷ, y) ≈ lossvalue
    @test logitcrossentropy(logylp, yl) ≈ -sum(yl.*logsoftmax(logylp))
    @test logitcrossentropy(logylp, yl, label_smoothing=2*ls) ≈ -sum(yls.*logsoftmax(logylp))
  end

  @testset "weighted_crossentropy" begin
    @test crossentropy(ŷ, y, weight = ones(2)) ≈ lossvalue
    @test crossentropy(ŷ, y, weight = [.5, .5]) ≈ lossvalue/2
    @test crossentropy(ylp, yl, weight = ones(2), label_smoothing=2ls) ≈ -sum(yls.*log.(ylp))
    @test crossentropy(ylp, yl, weight = [.5, .5], label_smoothing=2ls) ≈ -sum(yls.*log.(ylp))/2
    @test crossentropy(ŷ, y, weight = [2, .5]) ≈ 1.5049660054074199
  end

  @testset "weighted_logitcrossentropy" begin
    @test logitcrossentropy(logŷ, y, weight = ones(2)) ≈ lossvalue
    @test logitcrossentropy(logŷ, y, weight = [.5, .5]) ≈ lossvalue/2
    @test logitcrossentropy(logylp, yl, weight = ones(2), label_smoothing=0.2) ≈ -sum(yls.*logsoftmax(logylp))
    @test logitcrossentropy(logylp, yl, weight = [.5, .5],label_smoothing=0.2) ≈ -sum(yls.*logsoftmax(logylp))/2
    @test logitcrossentropy(logŷ, y, weight = [2, .5]) ≈ 1.5049660054074199
  end

  logŷ, y = randn(3), rand(3)
  yls = y.*(1-2ls).+ls

  @testset "binarycrossentropy" begin
    @test binarycrossentropy.(σ.(logŷ), y; ϵ=0, label_smoothing=2*ls) ≈ -yls.*log.(σ.(logŷ)) - (1 .- yls).*log.(1 .- σ.(logŷ))
    @test binarycrossentropy.(σ.(logŷ), y; ϵ=0) ≈ -y.*log.(σ.(logŷ)) - (1 .- y).*log.(1 .- σ.(logŷ))
    @test binarycrossentropy.(σ.(logŷ), y) ≈ -y.*log.(σ.(logŷ) .+ eps.(σ.(logŷ))) - (1 .- y).*log.(1 .- σ.(logŷ) .+ eps.(σ.(logŷ)))
  end

  @testset "logitbinarycrossentropy" begin
    @test logitbinarycrossentropy.(logŷ, y, label_smoothing=0.2) ≈ binarycrossentropy.(σ.(logŷ), y; ϵ=0, label_smoothing=0.2)
    @test logitbinarycrossentropy.(logŷ, y) ≈ binarycrossentropy.(σ.(logŷ), y; ϵ=0)
  end
  
  y = [1 2 3]
  y1 = [4.0 5.0 6.0]
  @testset "kldivergence" begin
    @test Flux.kldivergence(y, y1) ≈ 4.761838062403337
    @test Flux.kldivergence(y, y) ≈ 0 
  end
  
  y = [1 2 3 4]
  y1 = [5.0 6.0 7.0 8.0]
  @testset "hinge" begin
    @test Flux.hinge(y, y1) ≈ 0
    @test Flux.hinge(y, 0.5 .* y) ≈ 0.125
  end
  
  y = [0.1 0.2 0.3]
  y1 = [0.4 0.5 0.6]
  @testset "poisson" begin
    @test Flux.poisson(y, y1) ≈ 1.0160455586700767
    @test Flux.poisson(y, y) ≈ 0.5044459776946685
  end
  
  @testset "no spurious promotions" begin
    for T in (Float32, Float64)
      y = rand(T, 2)
      ŷ = rand(T, 2)
      for f in (mse, crossentropy, logitcrossentropy, Flux.kldivergence, Flux.hinge, Flux.poisson)
        fwd, back = Flux.pullback(f, ŷ, y)
        @test fwd isa T
        @test eltype(back(one(T))[1]) == T
      end
    end
  end
end
