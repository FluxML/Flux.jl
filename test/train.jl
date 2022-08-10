using Flux.Train
using Zygote: Params, gradient

import Optimisers, FillArrays, ComponentArrays, Yota

using Test
using Random

@testset "Implicit train!" begin  # These tests pass on Flux v0.13
  Random.seed!(84)
  w = randn(10, 10)
  w2 = randn(10, 10)  # NB outside the inner @testset, else it will be exactly == w, as the RNG seed is reset.
  @testset for opt in [AdamW(), AdaGrad(0.1), AdaMax(), AdaDelta(0.9), AMSGrad(),
                       NAdam(), RAdam(), Descent(0.1), Adam(), OAdam(), AdaBelief(),
                       Nesterov(), RMSProp(), Momentum()]
    w′ = copy(w2)
    b = zeros(10)
    loss(x) = Flux.Losses.mse(w*x, w′*x .+ b)
    @test loss(rand(10, 10)) > 1
    Flux.train!(loss, Flux.params([w′, b]), (rand(10) for _ in 1: 10^5), opt)
    @test loss(rand(10, 10)) < 0.01
  end
end

@testset "Explicit train! with Zygote" begin
  Random.seed!(84)
  w = randn(10, 10)
  w2 = randn(10, 10)  # NB outside the inner @testset, else it will be exactly == w, as the RNG seed is reset.
  @testset for opt in [AdamW(), AdaGrad(0.1), AdaMax(), AdaDelta(0.9), AMSGrad(),
                       NAdam(), RAdam(), Descent(0.1), Adam(), OAdam(), AdaBelief(),
                       Nesterov(), RMSProp(), Momentum()]
    @test opt isa FluxState
    @test opt.state isa Missing

    loss(m, x) = Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model, rand(10, 10)) > 1

    train!(loss, model, ((rand(10),) for _ in 1: 10^5), opt)
    @test loss(model, rand(10, 10)) < 0.01
    @test opt.state isa NamedTuple
  end
  
  # Test 3-arg `train!` method:
  @testset for opt in [Descent(0.1), Adam(), AdamW()]
    @test opt isa FluxState
    @test opt.state isa Missing

    loss(m) = let x = rand(10)
      Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    end
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model) > 1

    for i in 1:10^5
      train!(loss, model, opt)
    end
    @test loss(model) < 0.01
    @test opt.state isa NamedTuple
  end
  
  # Test direct use of Optimisers.jl rule, only really OK for `Descent`:
  @testset for opt in [Optimisers.Descent(0.1), Optimisers.Adam()]
    @test opt isa Optimisers.AbstractRule
    loss(m, x) = Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model, rand(10, 10)) > 1
    train!(loss, model, ((rand(10),) for _ in 1: 10^5), opt)
    @test loss(model, rand(10, 10)) < 0.01
  end
end

using Yota
using Flux: Descent, Adam, AdamW, FluxState
Flux.@train_autodiff Yota

@testset "Explicit train! with Yota" begin
  Random.seed!(84)
  w = randn(10, 10)
  w2 = randn(10, 10)  # NB outside the inner @testset, else it will be exactly == w, as the RNG seed is reset.
  @testset for opt in [Descent(0.1), Adam(), AdamW()]
    @test opt isa FluxState
    @test opt.state isa Missing

    loss(m, x) = Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model, rand(10, 10)) > 1

    train!(loss, model, ((rand(10),) for _ in 1: 10^5), opt)
    @test loss(model, rand(10, 10)) < 0.01
    @test opt.state isa NamedTuple
  end
  
  # Test 3-arg `train!` method:
  @testset for opt in [Descent(0.1), Adam(), AdamW()]
    @test opt isa FluxState
    @test opt.state isa Missing

    loss(m) = let x = rand(10)
      Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    end
    model = (weight=copy(w2), bias=zeros(10), ignore=nothing)
    @test loss(model) > 1

    for i in 1:10^5
      train!(loss, model, opt)
    end
    @test loss(model) < 0.01
    @test opt.state isa NamedTuple
  end
end

Flux.@train_autodiff Zygote

#=

@testset "update!: handle Fills from Zygote" begin
  w = randn(10,10)
  wold = copy(w)
  g = FillArrays.Ones(size(w))
  opt = Descent(0.1)
  Flux.update!(opt, w, g)
  @test w ≈ wold .- 0.1 

  w = randn(3)
  wold = copy(w)
  θ = Flux.params([w])
  gs = gradient(() -> w[1], θ)
  opt = Descent(0.1)
  Flux.update!(opt, θ, gs)
  @test w[1] ≈ wold[1] .- 0.1
  @test w[2:3] ≈ wold[2:3] 

  ## Issue #1510
  w = randn(10,10)
  wold = copy(w)
  θ = Flux.params([w])
  gs = gradient(() -> sum(w), θ)
  opt = Descent(0.1)
  Flux.update!(opt, θ, gs)
  @test w ≈ wold .- 0.1 
end

@testset "update!: handle ComponentArrays" begin
  w = ComponentArrays.ComponentArray(a=1.0, b=[2, 1, 4], c=(a=2, b=[1, 2]))
  wold = deepcopy(w)
  θ = Flux.params([w])
  gs = gradient(() -> sum(w.a) + sum(w.c.b), θ)
  opt = Descent(0.1)
  Flux.update!(opt, θ, gs)
  @test w.a ≈ wold.a .- 0.1
  @test w.b ≈ wold.b
  @test w.c.b ≈ wold.c.b .- 0.1
  @test w.c.a ≈ wold.c.a

  w = ComponentArrays.ComponentArray(a=1.0, b=[2, 1, 4], c=(a=2, b=[1, 2]))
  wold = deepcopy(w)
  θ = Flux.params([w])
  gs = gradient(() -> sum(w), θ)
  opt = Descent(0.1)
  Flux.update!(opt, θ, gs)
  @test w ≈ wold .- 0.1
end

=#