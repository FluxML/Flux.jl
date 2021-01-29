using Flux.Optimise
using Flux.Optimise: runall
using Flux: Params, gradient
using Test
using Random

@testset "Optimise" begin
  # Ensure rng has different state inside and outside the inner @testset
  # so that w and w' are different
  Random.seed!(84)
  w = randn(10, 10)
  @testset for opt in [ADAMW(), ADAGrad(η = 0.1), AdaMax(), ADADelta(ρ = 0.9), AMSGrad(),
                       NADAM(), RADAM(), Descent(η = 0.1), ADAM(), OADAM(), AdaBelief(),
                       Nesterov(), RMSProp(), Momentum()]
    Random.seed!(42)
    w′ = randn(10, 10)
    b = Flux.Zeros()
    θ = params([w′, b])
    st = init(opt, θ)
    loss(x) = Flux.Losses.mse(w*x, w′*x .+ b)
    for t = 1: 10^5
      x = rand(10)
      θ̄ = gradient(() -> loss(x), θ)
      θ, st = Optimise.update!(opt, θ, θ̄, st)
    end
    @test loss(rand(10, 10)) < 0.01
  end
end

@testset "Sequence" begin
  Random.seed!(84)
  w = randn(10, 10)
  w′ = randn(10, 10)
  loss(x) = Flux.Losses.mse(w*x, w′*x)
  θ = Params([w′])
  opt = Optimise.Sequence(WeightDecay(), ADAM(η = 0.001))
  st = init(opt, θ)
  for t = 1:10^5
    x = rand(10)
    θ̄ = gradient(() -> loss(x), θ)
    θ, st = Optimise.update!(opt, θ, θ̄, st)
  end
  @test loss(rand(10, 10)) < 0.01
end

@testset "Training Loop" begin
  i = 0
  l = 1
  Flux.train!(
              () -> (sleep(0.1); Flux.skip(); i+=1),
              (),
              Iterators.repeated((), 10),
              Descent()
             )

  @test i==0 #all skipped

  Flux.train!(
              () -> (sleep(0.1); i==8 && Flux.skip(); i+=1),
              (),
              Iterators.repeated((), 10),
              Descent()
             )

  @test i==8 #skip after i hit 8

  i = 0
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

  r = rand(3, 3)
  loss(x) = sum(x .* x)
  Flux.train!(loss, Flux.params(r), (r,), Descent())
end

# @testset "Clipping" begin
#     w = randn(10, 10)
#     loss(x) = sum(w * x)
#     θ = Params([w])
#     x = 1000 * randn(10)
#     w̄ = gradient(() -> loss(x), θ)[w]
#     w̄_value = Optimise.apply!(ClipValue(1.0), w, copy(w̄))
#     @test all(w̄_value .<= 1)
#     w̄_norm = Optimise.apply!(ClipNorm(1.0), w, copy(w̄))
#     @test norm(w̄_norm) <= 1
# end
