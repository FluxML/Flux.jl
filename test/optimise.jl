using Flux.Optimise
using Flux.Optimise: runall
using Flux.Tracker
using Test
@testset "Optimise" begin
  w = randn(10, 10)
  @testset for Opt in [ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, Descent, ADAM, Nesterov, RMSProp, Momentum]
    w′ = param(randn(10, 10))
    loss(x) = Flux.mse(w*x, w′*x)
    opt = Opt(0.001)
    if opt isa Descent || opt isa ADAGrad
      opt = Opt(0.1)
    end
    if opt isa ADADelta
      opt = Opt(0.9)
    end
    for t = 1: 10^5
      l = loss(rand(10))
      back!(l)
      delta = Optimise.update!(opt, w′.data, w′.grad)
      w′.data .-= delta
    end
    @test Flux.mse(w, w′) < 0.01
  end
end

@testset "Optimiser" begin
  w = randn(10, 10)
  @testset for Opt in [InvDecay, WeightDecay, ExpDecay]
    w′ = param(randn(10, 10))
    loss(x) = Flux.mse(w*x, w′*x)
    opt = Optimiser(Opt(), ADAM(0.001))
    for t = 1:10^5
      l = loss(rand(10))
      back!(l)
      delta = Optimise.update!(opt, w′.data, w′.grad)
      w′.data .-= delta
    end
    @test Flux.mse(w, w′) < 0.01
  end
end

@testset "Training Loop" begin
  i = 0
  l = param(1)

  Flux.train!(() -> (sleep(0.1); i += 1; l),
              (),
              Iterators.repeated((), 100),
              Descent(),
              cb = Flux.throttle(() -> (i > 3 && Flux.stop()), 1))

  @test 3 < i < 50

  # Test multiple callbacks
  x = 0
  fs = [() -> (), () -> x = 1]
  cbs = runall(fs)
  cbs()
  @test x == 1
end
