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

  # Test 3-arg `Flux.train!` method:
  @testset for rule in [Descent(0.1), Adam(), AdamW()]

    loss(m) = let x = rand(10)
      Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    end
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model) > 1

    opt = Flux.setup(rule, model)
    for i in 1:10^5
      Flux.train!(loss, model, opt)
    end
    @test loss(model) < 0.01
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
  # Test that splat accepts NamedTuple
  # Test NaN / Inf early stop
  # Test that loss is returned
end

import Tracker
Flux.@train_autodiff Tracker

@testset "Explicit Flux.train! with Tracker" begin
  Random.seed!(84)
  w = randn(10, 10)
  w2 = randn(10, 10)  # NB outside the inner @testset, else it will be exactly == w, as the RNG seed is reset.
  @testset for rule in [Descent(0.1), Adam(), AdamW()]

    loss(m, x) = begin
      Flux.istraining() && error("This test is not in fact using Tracker!")
      Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    end
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model, rand(10, 10)) > 1

    opt = Flux.setup(rule, model)
    Flux.train!(loss, model, ((rand(10),) for _ in 1: 10^5), opt)
    @test loss(model, rand(10, 10)) < 0.01
  end

  # Test 3-arg `Flux.train!` method:
  @testset for rule in [Descent(0.1), Adam()]

    loss(m) = let x = rand(10)
      Flux.istraining() && error("This test is not in fact using Tracker!")
      Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    end
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model) > 1

    opt = Flux.setup(rule, model)
    for i in 1:10^5
      Flux.train!(loss, model, opt)
    end
    @test loss(model) < 0.01
  end
end

import Yota
Flux.@train_autodiff Yota

@testset "Explicit Flux.train! with Yota" begin
  Random.seed!(84)
  w = randn(10, 10)
  w2 = randn(10, 10)  # NB outside the inner @testset, else it will be exactly == w, as the RNG seed is reset.
  @testset for rule in [Descent(0.1), Adam(), AdamW()]

    loss(m, x) = Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model, rand(10, 10)) > 1

    opt = Flux.setup(rule, model)
    Flux.train!(loss, model, ((rand(10),) for _ in 1: 10^5), opt)
    @test loss(model, rand(10, 10)) < 0.01
  end

  # Test 3-arg `Flux.train!` method:
  @testset for rule in [Descent(0.1), Adam()]

    loss(m) = let x = rand(10)
      Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    end
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model) > 1

    opt = Flux.setup(rule, model)
    for i in 1:10^5
      Flux.train!(loss, model, opt)
    end
    @test loss(model) < 0.01
  end
end

Flux.@train_autodiff Zygote
