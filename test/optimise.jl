using Flux.Optimise
using Flux.Tracker

@testset "Optimise" begin
  w = randn(10, 10)
  for Opt in [SGD, Nesterov, Momentum, ADAM, RMSProp, ps -> ADAGrad(ps, 0.1), ADADelta, AMSGrad]
    w′ = param(randn(10, 10))
    loss(x) = Flux.mse(w*x, w′*x)
    opt = Opt([w′])
    for t=1:10^5
      l = loss(rand(10))
      back!(l)
      opt()
    end
    @test Flux.mse(w, w′) < 0.01
  end
end

@testset "Training Loop" begin
  l = param(1)
  loss_calls = 0
  log_calls = 0
  opt_calls = 0
  Flux.train!(() -> (sleep(0.1); loss_calls+=1; l),
              Iterators.repeated((), 100),
              ()->(opt_calls+=1; nothing),
              log_cb = Flux.throttle((j,v) -> log_calls+=1, 1),
              stopping_criteria = Flux.throttle((j,v) -> (j > 3), 1))

  @test 3 < loss_calls < 50
  @test log_calls == 2
  @test opt_calls == loss_calls - 1
end


@testset "Trivial Training Loop" begin
  l = param(1)
  Flux.train!(() -> l,
              Iterators.repeated((), 100),
              ()->())
  # all it has to do is not error
end
