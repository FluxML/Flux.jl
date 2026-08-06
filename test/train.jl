
function train_enzyme!(fn, model, args...; kwargs...)
  Flux.train!(fn, Enzyme.Duplicated(model, Enzyme.make_zero(model)), args...; kwargs...)
end

for (trainfn!, name) in ((Flux.train!, "Zygote"), (train_enzyme!, "Enzyme"))

  if name == "Enzyme" && !FLUX_TEST_ENZYME
    continue
  end

  @testset "Flux.train! with $name" begin
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
      trainfn!(loss, model, ((rand(10),) for _ in 1: 10^5), opt)
      @test loss(model, rand(10, 10)) < 0.01
    end
  end
end

for (trainfn!, name) in ((Flux.train!, "Zygote"), (train_enzyme!, "Enzyme"))
  if name == "Enzyme" && !FLUX_TEST_ENZYME
    continue
  end

  @testset "Flux.train! features with $name" begin
    @testset "Stop on NaN" begin
      m1 = Dense(1 => 1)
      m1.weight .= 0
      CNT = Ref(0)
      @test_throws DomainError trainfn!(m1, tuple.(1:100), Descent(0.1)) do m, i
        CNT[] += 1
        (i == 51 ? NaN32 : 1f0) * sum(m([1f0]))
      end
      @test CNT[] == 51  # stopped early
      @test m1.weight[1] ≈ -5  # did not corrupt weights
    end

    @testset "non-tuple data" begin
      w = randn(10, 10)
      w2 = randn(10, 10)
      loss(m, x) = Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
      model = (weight=copy(w2), bias=zeros(10))
      opt = Flux.setup(AdamW(), model)
      trainfn!(loss, model, (rand(10) for _ in 1: 10^5), opt)
      @test loss(model, rand(10, 10)) < 0.01
    end

    @testset "callbacks give helpful error" begin
      m1 = Dense(1 => 1)
      cb = () -> println("this should not be printed")
      @test_throws ErrorException trainfn!((args...,) -> 1, m1, [(1,2)], Descent(0.1); cb)
    end
  end
end

for name in ("Zygote", "Enzyme")
  if name == "Enzyme" && !FLUX_TEST_ENZYME
    continue
  end

  # For Enzyme the model is passed to `trainstep!` wrapped in a `Duplicated` (whose `.val` is the
  # same object as `model`, so `model` is still what gets mutated in place); for Zygote it's passed
  # as-is.
  wrap = name == "Enzyme" ? (m -> Enzyme.Duplicated(m, Enzyme.make_zero(m))) : identity

  @testset "Flux.trainstep! with $name" begin
    @testset "trainstep! returns the loss and updates in place" begin
      model = Dense(3 => 2, tanh)
      x, y = randn(Float32, 3, 5), randn(Float32, 2, 5)
      loss(m, x, y) = Flux.mse(m(x), y)

      wm = wrap(model)
      opt = Flux.setup(Momentum(0.1), wm)
      l0 = loss(model, x, y)
      w0 = copy(model.weight)
      vel0 = copy(opt.weight.state)   # Momentum velocity, initially zero

      l = Flux.trainstep!(loss, wm, (x, y), opt)

      @test l isa Real && isfinite(l)
      @test l ≈ l0                          # loss is measured *before* the update
      @test !(model.weight ≈ w0)            # model mutated in place
      @test !(opt.weight.state ≈ vel0)      # optimiser state mutated in place
    end

    @testset "trainstep_withgradient! also returns the gradient" begin
      model = Dense(3 => 2, tanh)
      x, y = randn(Float32, 3, 5), randn(Float32, 2, 5)
      loss(m, x, y) = Flux.mse(m(x), y)

      wm = wrap(model)
      opt = Flux.setup(Momentum(0.1), wm)
      l0 = loss(model, x, y)
      w0 = copy(model.weight)

      l, g = Flux.trainstep_withgradient!(loss, wm, (x, y), opt)

      @test l isa Real && isfinite(l)
      @test l ≈ l0                          # loss is measured *before* the update
      @test g.weight isa AbstractArray && size(g.weight) == size(model.weight)
      @test !(model.weight ≈ w0)            # model still mutated in place
    end

    @testset "repeated steps converge" begin
      Random.seed!(84)
      w = randn(10, 10)
      loss(m, x) = Flux.Losses.mse(w*x, m.weight*x .+ m.bias)
      model = (weight=randn(10, 10), bias=zeros(10), ignore=nothing)
      @test loss(model, rand(10, 10)) > 1

      wm = wrap(model)
      opt = Flux.setup(Adam(0.05), wm)
      for _ in 1:2000
        Flux.trainstep!(loss, wm, (rand(10, 32),), opt)
      end
      @test loss(model, rand(10, 10)) < 0.01
    end

    @testset "non-finite loss skips the update" begin
      model = Dense(1 => 1)
      model.weight .= 0
      model.bias .= 0
      wm = wrap(model)
      opt = Flux.setup(Descent(0.1), wm)

      l = Flux.trainstep!((m, i) -> NaN32 * sum(m([1f0])), wm, (1,), opt)
      @test !isfinite(l)
      @test model.weight ≈ [0f0;;]   # update skipped, model left uncorrupted
      @test model.bias ≈ [0f0]
    end

    @testset "auxiliary loss outputs" begin
      x, y = randn(Float32, 3, 5), randn(Float32, 2, 5)
      mse(m, x, y) = Flux.mse(m(x), y)
      tuploss(m, x, y) = (Flux.mse(m(x), y), m(x))               # aux: a Tuple with an array
      ntloss(m, x, y)  = (loss=Flux.mse(m(x), y), pred=m(x))     # aux: a NamedTuple

      # Aux is returned verbatim, and differentiating `first∘loss` leaves the update identical to the
      # plain scalar-loss run (aux does not perturb the gradient).
      base = Dense(3 => 2, tanh)
      l0, pred0 = Flux.mse(base(x), y), base(x)                  # pre-update loss and prediction
      for auxfn in (tuploss, ntloss)
        model = deepcopy(base); ref = deepcopy(base)
        wm, wref = wrap(model), wrap(ref)
        opt  = Flux.setup(Momentum(0.1), wm)
        optr = Flux.setup(Momentum(0.1), wref)

        v = Flux.trainstep!(auxfn, wm, (x, y), opt)
        Flux.trainstep!(mse, wref, (x, y), optr)                # same step, scalar loss

        @test v isa Union{Tuple, NamedTuple}                    # full value returned
        @test first(v) ≈ l0                                     # scalar loss, measured pre-update
        @test v[2] ≈ pred0                                      # aux is the pre-update prediction
        @test model.weight ≈ ref.weight                        # identical update to the scalar run
      end

      # trainstep_withgradient! returns ((loss, aux...), grad)
      model = deepcopy(base); wm = wrap(model)
      opt = Flux.setup(Momentum(0.1), wm)
      v, g = Flux.trainstep_withgradient!(ntloss, wm, (x, y), opt)
      @test v.loss ≈ l0
      @test v.pred ≈ pred0
      @test g.weight isa AbstractArray && size(g.weight) == size(model.weight)
    end
  end
end

@testset "Flux.train! is a loop over Flux.trainstep!" begin
  Random.seed!(123)
  data = [(randn(Float32, 3, 5), randn(Float32, 2, 5)) for _ in 1:20]
  loss(m, x, y) = Flux.mse(m(x), y)

  m1 = Dense(3 => 2, tanh)
  m2 = deepcopy(m1)
  o1 = Flux.setup(Adam(0.05), m1)
  o2 = Flux.setup(Adam(0.05), m2)

  Flux.train!(loss, m1, data, o1)
  for (x, y) in data
    Flux.trainstep!(loss, m2, (x, y), o2)
  end

  @test m1.weight ≈ m2.weight
  @test m1.bias ≈ m2.bias
end

@testset "Flux.update! features" begin
  m = Chain(Dense(2=>3, tanh), Dense(3=>1), only)
  x = rand(Float32, 2)
  y1 = m(x)  # before

  # Explicit gradient
  gs = Zygote.gradient(marg -> marg(x), m)
  @test gs isa Tuple
  @test_throws ErrorException Flux.update!(Flux.Adam(), m, gs)  # friendly
  @test_throws ErrorException Flux.update!(Flux.Adam(), m, gs[1])  # friendly
  s = Flux.setup(Adam(), m)
  @info "ignore this warning, just testing an upgrade path:"
  Flux.update!(s, m, gs)  # Chain + Tuple can be unambiguously sorted out
  y2 = m(x)
  @test y2 < y1
  Flux.update!(s, m, gs[1])  # finally, this is the correct thing
  y3 = m(x)
  @test y3 < y2

  # Also check that if you import the new Adam, then Flux.setup does still work!
  s2 = Flux.setup(Optimisers.Adam(), m)
  Flux.update!(s2, m, gs[1])
  y4 = m(x)
  @test y4 < y3
end

for (trainfn!, name) in ((Flux.train!, "Zygote"), (train_enzyme!, "Enzyme"))

  if name == "Enzyme" && !FLUX_TEST_ENZYME
    continue
  end
  
  @testset "L2 regularisation with $name" begin
    # New docs claim an exact equivalent. It's a bit long to put the example in there,
    # but perhaps the tests should contain it.

    model = Dense(3 => 2, tanh);
    init_weight = copy(model.weight);
    data = [(randn(Float32, 3,5), randn(Float32, 2,5)) for _ in 1:10];

    # Take 1: explicitly add a penalty in the loss function
    opt = Flux.setup(Adam(0.1), model)
    trainfn!(model, data, opt) do m, x, y
      err = Flux.mse(m(x), y)
      l2 = sum(abs2, m.weight)/2 + sum(abs2, m.bias)/2
      err + 0.33 * l2
    end
    diff1 = model.weight .- init_weight

    # Take 2: the same, but with Optimisers.trainables. 
    model.weight .= init_weight
    model.bias .= 0
    pen2(x::AbstractArray) = sum(abs2, x)/2
    opt = Flux.setup(Adam(0.1), model)

    trainfn!(model, data, opt) do m, x, y
      err = Flux.mse(m(x), y)
      l2 = sum(pen2, Flux.trainables(m))
      err + 0.33 * l2
    end
    

    diff2 = model.weight .- init_weight
    @test diff1 ≈ diff2

    # Take 3: using WeightDecay instead. Need the /2 above, to match exactly.
    model.weight .= init_weight
    model.bias .= 0
    decay_opt = Flux.setup(OptimiserChain(WeightDecay(0.33), Adam(0.1)), model);
    trainfn!(model, data, decay_opt) do m, x, y
      Flux.mse(m(x), y)
    end
    diff3 = model.weight .- init_weight
    @test diff1 ≈ diff3
  end
end

@testset "Flux.setup bugs" begin
  # https://github.com/FluxML/Flux.jl/issues/2144
  @test Flux.setup(Flux.Adam(), Embedding(3 => 1)).weight isa Optimisers.Leaf
  
  @test Flux.setup(Flux.ClipGrad(1), Dense(2 => 3)).weight.rule isa Optimisers.ClipGrad
  @test Flux.setup(Flux.ClipNorm(1), Dense(2 => 3)).weight.rule isa Optimisers.ClipNorm
end
