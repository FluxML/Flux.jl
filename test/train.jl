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
    m1 = Dense(1 => 1)
    m1.weight .= 0
    CNT = 0
    @test_throws DomainError Flux.train!(m1, tuple.(1:100), Descent(0.1)) do m, i
      CNT += 1
      (i == 51 ? NaN32 : 1f0) * sum(m([1.0]))
    end
    @test CNT == 51  # stopped early
    @test m1.weight[1] ≈ -5  # did not corrupt weights
  end

  @testset "non-tuple data" begin
    w = randn(10, 10)
    w2 = randn(10, 10)
    loss(m, x) = Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    model = (weight=copy(w2), bias=zeros(10))
    opt = Flux.setup(AdamW(), model)
    Flux.train!(loss, model, (rand(10) for _ in 1: 10^5), opt)
    @test loss(model, rand(10, 10)) < 0.01
  end

  @testset "callbacks give helpful error" begin
    m1 = Dense(1 => 1)
    cb = () -> println("this should not be printed")
    @test_throws ErrorException Flux.train!((args...,) -> 1, m1, [(1,2)], Descent(0.1); cb)
  end
end

@testset "Explicit Flux.update! features" begin
  m = Chain(Dense(2=>3, tanh), Dense(3=>1), only)
  x = rand(2)
  y1 = m(x)  # before

  # Implicit gradient
  gold = gradient(() -> m(x), Flux.params(m))
  @test gold isa Flux.Zygote.Grads
  @test_throws ErrorException Flux.update!(Flux.Adam(), m, gold)  # friendly
  Flux.update!(Flux.Adam(), Flux.params(m), gold)
  y2 = m(x)
  @test y2 < y1

  # Explicit gradient
  gs = gradient(marg -> marg(x), m)
  @test gs isa Tuple
  @test_throws ErrorException Flux.update!(Flux.Adam(), Flux.params(m), gs) # friendly
  @test_throws ErrorException Flux.update!(Flux.Adam(), Flux.params(m), gs[1]) # friendly
  @test_throws ErrorException Flux.update!(Flux.Adam(), m, gs)  # friendly
  @test_throws ErrorException Flux.update!(Flux.Adam(), m, gs[1])  # friendly
  s = Flux.setup(Adam(), m)
  @info "ignore this warning, just testing an upgrade path:"
  Flux.update!(s, m, gs)  # Chain + Tuple can be unambiguously sorted out
  y3 = m(x)
  @test y3 < y2
  Flux.update!(s, m, gs[1])  # finally, this is the correct thing
  y4 = m(x)
  @test y4 < y3

  # Also check that if you import the new Adam, then Flux.setup does still work!
  s2 = Flux.setup(Optimisers.Adam(), m)
  Flux.update!(s2, m, gs[1])
  y5 = m(x)
  @test y5 < y4
end

@testset "L2 regularisation" begin
  # New docs claim an exact equivalent. It's a bit long to put the example in there,
  # but perhaps the tests should contain it.

  model = Dense(3 => 2, tanh);
  init_weight = copy(model.weight);
  data = [(randn(Float32, 3,5), randn(Float32, 2,5)) for _ in 1:10];

  # Take 1: explicitly add a penalty in the loss function
  opt = Flux.setup(Adam(0.1), model)
  Flux.train!(model, data, opt) do m, x, y
    err = Flux.mse(m(x), y)
    l2 = sum(abs2, m.weight)/2 + sum(abs2, m.bias)/2
    err + 0.33 * l2
  end
  diff1 = model.weight .- init_weight

  # Take 2: the same, but with Flux.params. Was broken for a bit, no tests!
  model.weight .= init_weight
  model.bias .= 0
  pen2(x::AbstractArray) = sum(abs2, x)/2
  opt = Flux.setup(Adam(0.1), model)
  Flux.train!(model, data, opt) do m, x, y
    err = Flux.mse(m(x), y)
    l2 = sum(pen2, Flux.params(m))
    err + 0.33 * l2
  end
  diff2 = model.weight .- init_weight
  @test diff1 ≈ diff2

  # Take 3: using WeightDecay instead. Need the /2 above, to match exactly.
  model.weight .= init_weight
  model.bias .= 0
  decay_opt = Flux.setup(OptimiserChain(WeightDecay(0.33), Adam(0.1)), model);
  Flux.train!(model, data, decay_opt) do m, x, y
    Flux.mse(m(x), y)
  end
  diff3 = model.weight .- init_weight
  @test diff1 ≈ diff3
end

