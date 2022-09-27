using Flux.Optimise
using Flux.Optimise: runall
using Flux: Params, gradient
import FillArrays, ComponentArrays
using Test
using Random

@testset "Optimise" begin
  # Ensure rng has different state inside and outside the inner @testset
  # so that w and w' are different
  Random.seed!(84)
  w = randn(10, 10)
  @testset for opt in [AdamW(), AdaGrad(0.1), AdaMax(), AdaDelta(0.9), AMSGrad(),
                       NAdam(), RAdam(), Descent(0.1), Adam(), OAdam(), AdaBelief(),
                       Nesterov(), RMSProp(), Momentum()]
    Random.seed!(42)
    w′ = randn(10, 10)
    b = false
    loss(x) = Flux.Losses.mse(w*x, w′*x .+ b)
    for t = 1: 10^5
      θ = params([w′, b])
      x = rand(10)
      θ̄ = gradient(() -> loss(x), θ)
      Optimise.update!(opt, θ, θ̄)
    end
    @test loss(rand(10, 10)) < 0.01
  end
end

@testset "Optimiser" begin
  Random.seed!(84)
  w = randn(10, 10)
  @testset for Opt in [InvDecay, WeightDecay, ExpDecay]
    Random.seed!(42)
    w′ = randn(10, 10)
    loss(x) = Flux.Losses.mse(w*x, w′*x)
    opt = Optimiser(Opt(), Adam(0.001))
    for t = 1:10^5
      θ = Params([w′])
      x = rand(10)
      θ̄ = gradient(() -> loss(x), θ)
      Optimise.update!(opt, θ, θ̄)
    end
    @test loss(rand(10, 10)) < 0.01
  end
end

@testset "Training Loop" begin
  i = 0
  l = 1
  Flux.train!(
              () -> (sleep(0.1); Flux.skip(); i+=1),
              Params([]),
              Iterators.repeated((), 10),
              Descent()
             )

  @test i==0 #all skipped

  Flux.train!(
              () -> (sleep(0.1); i==8 && Flux.skip(); i+=1),
              Params([]),
              Iterators.repeated((), 10),
              Descent()
             )

  @test i==8 #skip after i hit 8

  i = 0
  Flux.train!(() -> (sleep(0.1); i += 1; l),
              Params([]),
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

@testset "Stop on NaN" begin
  m = Dense(1 => 1)
  m.weight .= 0
  CNT = 0
  Flux.train!(Flux.params(m), 1:100, Descent(0.1)) do i
    CNT += 1
    (i == 51 ? NaN32 : 1f0) * sum(m([1.0]))
  end
  @test CNT == 51  # stopped early
  @test m.weight[1] ≈ -5  # did not corrupt weights
end

@testset "ExpDecay" begin

  @testset "Sanity Check" begin
    o = ExpDecay(0.2, 0.5, 1, 1e-3)
    p = [0.0]
    steps = 1:8
    eta_expected = @. max(o.eta * 0.5 ^ steps, o.clip)
    eta_actual = [Optimise.apply!(o, p, [1.0])[1] for _ in steps]
    @test eta_actual == eta_expected
  end

  @testset "starting step" begin
    start = 4
    o = ExpDecay(0.2, 0.5, 1, 1e-3, start)
    p = [0.0]
    steps = 1:8
    eta_expected = @. max(o.eta * 0.5 ^ max(steps - start, 0), o.clip)
    eta_actual = [Optimise.apply!(o, p, [1.0])[1] for _ in steps]
    @test eta_actual == eta_expected
  end

  w = randn(10, 10)
  o = ExpDecay(0.1, 0.1, 1000, 1e-4)
  w1 = randn(10,10)
  loss(x) = Flux.Losses.mse(w*x, w1*x)
  flag = 1
  decay_steps = []
  for t = 1:10^5
    prev_eta = o.eta
    θ = Params([w1])
    x = rand(10)
    θ̄ = gradient(() -> loss(x), θ)
    prev_grad = collect(θ̄[w1])
    delta = Optimise.apply!(o, w1, θ̄[w1])
    w1 .-= delta
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
  # Test to check if decay happens at decay steps. Eta reaches clip value (1e-4) after 4000 steps (decay by 0.1 every 1000 steps starting at 0.1).
  ground_truth = []
  for i in 1:4
    push!(ground_truth, 1000*i)  # Expected decay steps for this example.
  end
  @test decay_steps == ground_truth
  @test o.eta == o.clip
end

@testset "Clipping" begin
  w = randn(10, 10)
  loss(x) = sum(w * x)
  θ = Params([w])
  x = 1000 * randn(10)
  w̄ = gradient(() -> loss(x), θ)[w]
  w̄_value = Optimise.apply!(ClipValue(1.0), w, copy(w̄))
  @test all(w̄_value .<= 1)
  w̄_norm = Optimise.apply!(ClipNorm(1.0), w, copy(w̄))
  @test norm(w̄_norm) <= 1
end

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

# Flux PR #1776
# We need to test that optimisers like Adam that maintain an internal momentum
# estimate properly calculate the second-order statistics on the gradients as
# the flow backward through the model.  Previously, we would calculate second-
# order statistics via `Δ^2` rather than the complex-aware `Δ * conj(Δ)`, which
# wreaks all sorts of havoc on our training loops.  This test ensures that
# a simple optimization is montonically decreasing (up to learning step effects)
@testset "Momentum Optimisers and complex values" begin
  # Test every optimizer that has momentum internally
  for opt_ctor in [Adam, RMSProp, RAdam, OAdam, AdaGrad, AdaDelta, NAdam, AdaBelief]
    # Our "model" is just a complex number
    w = zeros(ComplexF32, 1)

    # Our model attempts to learn `f(x) = conj(x)` where `f(x) = w*x`
    function loss()
        # Deterministic training data is the best training data
        x = ones(1, 1) + 1im*ones(1, 1)

        # Manually implement `mse()` to allow demonstration of brokenness
        # on older Flux builds that don't have a fixed `mse()`
        return sum(abs2.(w * x .- conj(x)))
    end

    params = Flux.Params([w])
    opt = opt_ctor(1e-2)

    # Train for 10 iterations, enforcing that loss is monotonically decreasing
    last_loss = Inf
    for idx in 1:10
        grads = Flux.gradient(loss, params)
        @test loss() < last_loss
        last_loss = loss()
        Flux.update!(opt, params, grads)
    end
  end
end
