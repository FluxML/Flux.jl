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
