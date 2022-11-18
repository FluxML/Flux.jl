using Flux
# using Flux.Train
import Optimisers

using Test
using Random

@testset "Explicit Flux.train! with Zygote" begin
  Random.seed!(84)
  w = randn(10, 10)
  w2 = randn(10, 10)  # NB outside the inner @testset, else it will be exactly == w, as the RNG seed is reset.
  @testset for rule in [AdamW(), AdaGrad(0.1), AdaMax(), AdaDelta(0.9), AMSGrad(),
                        NAdam(), RAdam(), Descent(0.1), Adam(), OAdam(), AdaBelief(),
                        Nesterov(), RMSProp(), Momentum()]

    loss(m, x) = Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model, rand(10, 10)) > 1

    opt = Flux.setup(rule, model)
    Flux.train!(loss, model, ((rand(10),) for _ in 1: 10^5), opt)
    @test loss(model, rand(10, 10)) < 0.01
  end

  # Test direct use of Optimisers.jl rule, only really OK for `Descent`:
  @testset "without setup, $opt" for opt in [Descent(0.1), Optimisers.Descent(0.1), Optimisers.Adam()]
    loss(m, x) = Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model, rand(10, 10)) > 1
    Flux.train!(loss, model, ((rand(10),) for _ in 1: 10^5), opt)
    @test loss(model, rand(10, 10)) < 0.01
  end
end

@testset "Explicit Flux.train! features" begin
  @testset "Stop on NaN" begin
    m = Dense(1 => 1)
    m.weight .= 0
    CNT = 0
    @test_throws DomainError Flux.train!(m, tuple.(1:100), Descent(0.1)) do (i,)
      CNT += 1
      (i == 51 ? NaN32 : 1f0) * sum(m([1.0]))
    end
    @test CNT == 51  # stopped early
    @test m.weight[1] ≈ -5  # did not corrupt weights
  end
  @testset "data must give tuples" begin
    m = Dense(1 => 1)
    @test_throws ErrorException Flux.train!((args...,) -> 1, m, [(x=1, y=2) for _ in 1:3], Descent(0.1))
  end
end
