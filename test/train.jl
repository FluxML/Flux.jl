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

  @testset "non-tuple data" begin
    loss(m, x) = Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
    model = (weight=copy(w2), bias=zeros(10))
    opt = Flux.setup(AdamW(), model)
    Flux.train!(loss, model, (rand(10) for _ in 1: 10^5), opt)
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

  @testset "deprecated callback style" begin
    m1 = Dense(1 => 1)
    cb = () -> println("this should not be printed")
    Flux.train!((args...,) -> 1, m1, [(1,2)], Descent(0.1); cb)
  end


  @testset "callback" begin
    m1 = Dense(1 => 1)
    i = 0 
    data = [rand(1) for _ in 1:5]
    res = []
    cb = x -> push!(res, x)
    opt = Flux.setup(AdamW(), m1)
    Flux.train!((m, x) -> sum(m(x)), m1, data, opt; cb)

    @test length(res) == length(data)
    for (i,x) in enumerate(res)
      @test x isa NamedTuple
      @test x.step == i
      @test haskey(x, :loss)
      @test x.gradient.weight isa Matrix
      @test x.gradient.bias isa Vector
      @test x.model === m1
      @test haskey(x, :data)
      @test x.opt === opt
    end
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

