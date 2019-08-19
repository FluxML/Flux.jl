using Flux.Optimise
using Flux.Optimise: runall
using Flux.Tracker
using Test
@testset "Optimise" begin
  w = randn(10, 10)
  @testset for opt in [ADAMW(), ADAGrad(0.1), AdaMax(), ADADelta(0.9), AMSGrad(),
                       NADAM(), RADAM(), Descent(0.1), ADAM(), Nesterov(), RMSProp(),
                       Momentum()]
    w′ = param(randn(10, 10))
    loss(x) = Flux.mse(w*x, w′*x)
    for t = 1: 10^5
      θ = Params([w′])
      θ̄ = gradient(() -> loss(rand(10)), θ)
      Optimise.update!(opt, θ, θ̄)
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
      delta = Optimise.apply!(opt, w′.data, w′.grad)
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

@testset "ExpDecay" begin
    w = randn(10, 10)
    o = ExpDecay(0.1, 0.1, 1000, 1e-4)
    w1 = param(randn(10,10))
    loss(x) = Flux.mse(w*x, w1*x)
    flag = 1
    decay_steps = []
    for t = 1:10^5
      l = loss(rand(10))
      back!(l)
      prev_eta = o.eta
      prev_grad = collect(w1.grad)
      delta = Optimise.apply!(o, w1.data, w1.grad)
      w1.data .-= delta
      new_eta = o.eta
      if new_eta != prev_eta
        push!(decay_steps, t)
      end
      array = fill(o.eta, size(prev_grad))
      if array .* prev_grad != delta
        flag = 0
      end
    end
    @test flag == 1
    # Test to check if decay happens at decay steps. Eta reaches clip value eventually.
    ground_truth = []
    for i in 1:11
      push!(ground_truth, 1000*i)  # Expected decay steps for this example.
    end
    @test decay_steps == ground_truth
    @test o.eta == o.clip
end
