using Flux.Optimise
using Flux.Tracker
using Test
@testset "Optimise" begin
  w = randn(10, 10)
  @testset for Opt in [Descent, ADAM, Nesterov, RMSProp, Momentum]
    w′ = param(randn(10, 10))
    loss(x) = Flux.mse(w*x, w′*x)
    opt = Opt(0.001)
    if opt isa Descent
      opt = Opt(0.1)
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

@testset "Training Loop" begin
  i = 0
  l = param(1)

  Flux.train!(() -> (sleep(0.1); i += 1; l),
              Iterators.repeated((), 100),
              ()->(),
              cb = Flux.throttle(() -> (i > 3 && Flux.stop()), 1))

  @test 3 < i < 50
end
