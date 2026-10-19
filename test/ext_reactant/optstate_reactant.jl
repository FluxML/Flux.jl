# Resuming training on a Reactant device from an optimiser state saved on the host (#2744).

@testset "resume a host optimiser state on Reactant" begin
    dev = MLDataDevices.reactant_device(force=true)
    loss(m, x, y) = Flux.mse(m(x), y)
    model0 = Chain(Dense(4 => 8, tanh), Dense(8 => 2))
    batches = [(randn(Float32, 4, 16), randn(Float32, 2, 16)) for _ in 1:10]

    # Not `ClipNorm`, which Optimisers 0.4.10 can't `adjust!` on Reactant.
    @testset "$rule" for rule in (Adam(0.01f0), AdamW(0.01f0, (0.9, 0.999), 0.01f0), Momentum(0.01f0),
                                  RAdam(0.01f0), OptimiserChain(ClipGrad(0.1f0), Adam(0.01f0)))
        eager = deepcopy(model0)
        opt_eager = Flux.setup(rule, eager)
        for b in batches[1:5]
            Flux.trainstep!(loss, eager, b, opt_eager)
        end
        # A checkpoint, as saved with JLD2 and loaded back.
        model_state, saved_opt = deepcopy(Flux.state(eager)), deepcopy(opt_eager)
        for b in batches[6:10]
            Flux.trainstep!(loss, eager, b, opt_eager)
        end

        model = Flux.loadmodel!(deepcopy(model0), model_state) |> dev
        opt = Flux.setup(rule, model)
        @test Flux.loadmodel!(opt, saved_opt) === opt
        for b in batches[6:10]
            Flux.trainstep!(loss, model, b |> dev, opt)
        end
        # `rtol` allows for CUDA's TF32-like matmuls; losing Adam's βt is off by 0.3.
        for (p, q) in zip(Flux.trainables(model |> cpu), Flux.trainables(eager))
            @test p ≈ q rtol=1e-3
        end

        # The rule from `setup` keeps its hyperparameters on the device, so `adjust!` still applies.
        Flux.adjust!(opt_eager, 0.1f0)
        Flux.adjust!(opt, 0.1f0)
        for b in batches[1:5]
            Flux.trainstep!(loss, eager, b, opt_eager)
            Flux.trainstep!(loss, model, b |> dev, opt)
        end
        for (p, q) in zip(Flux.trainables(model |> cpu), Flux.trainables(eager))
            @test p ≈ q rtol=1e-3
        end
    end
end

@testset "load optimiser state between host and Reactant" begin
    dev = MLDataDevices.reactant_device(force=true)
    loss(m, x, y) = Flux.mse(m(x), y)
    model0 = Chain(Dense(4 => 8, tanh), Dense(8 => 2))
    batches = [(randn(Float32, 4, 16), randn(Float32, 2, 16)) for _ in 1:10]

    # RAdam's step counter is an `Int` on the host, a tracked `Float32` on the device.
    @testset "$rule" for rule in (Adam(0.01f0), RAdam(0.01f0))
        eager = deepcopy(model0)
        opt_eager = Flux.setup(rule, eager)
        for b in batches
            Flux.trainstep!(loss, eager, b, opt_eager)
        end

        model = model0 |> dev
        opt = Flux.setup(rule, model)
        for b in batches[1:5]
            Flux.trainstep!(loss, model, b |> dev, opt)
        end
        # A checkpoint saved from the device, then resumed on the device and on the host.
        model_state, opt_state = cpu(Flux.state(model)), cpu(Flux.state(opt))

        model2 = Flux.loadmodel!(deepcopy(model0), model_state) |> dev
        opt2 = Flux.setup(rule, model2)
        Flux.loadmodel!(opt2, opt_state)
        model3 = Flux.loadmodel!(deepcopy(model0), model_state)
        opt3 = Flux.setup(rule, model3)
        Flux.loadmodel!(opt3, opt)
        @test typeof(opt3) == typeof(opt_eager)
        for b in batches[6:10]
            Flux.trainstep!(loss, model2, b |> dev, opt2)
            Flux.trainstep!(loss, model3, b, opt3)
        end
        for (p, q, r) in zip(Flux.trainables(model2 |> cpu), Flux.trainables(model3), Flux.trainables(eager))
            @test p ≈ r rtol=1e-3
            @test q ≈ r rtol=1e-3
        end
    end
end

@testset "loaded optimiser scalars stay on the model's client" begin
    # Not the default client when a GPU is available.
    cpuc = Reactant.XLA.client("cpu")
    dev = MLDataDevices.reactant_device(; client=cpuc, device=Reactant.XLA.default_device(cpuc))
    opt = Flux.setup(Adam(0.01f0), Dense(2 => 2) |> dev)
    Flux.loadmodel!(opt, Flux.setup(Adam(0.01f0), Dense(2 => 2)))
    @test all(βt -> Reactant.XLA.client(βt) == cpuc, opt.weight.state[3])
end

@testset "host optimiser state rejected on Reactant" begin
    dev = MLDataDevices.reactant_device(force=true)
    loss(m, x, y) = Flux.mse(m(x), y)
    model0 = Chain(Dense(4 => 8, tanh), Dense(8 => 2))
    model = model0 |> dev
    x, y = randn(Float32, 4, 16) |> dev, randn(Float32, 2, 16) |> dev

    # Adam's βt would fail to compile, and Descent's learning rate would be baked into the
    # executable, ignoring later `adjust!` calls.
    for opt in (Flux.setup(Adam(0.01f0), model0) |> dev, Flux.setup(Descent(0.1f0), model0) |> dev)
        @test_throws "loadmodel!" Flux.trainstep!(loss, model, (x, y), opt)
        @test_throws ArgumentError Flux.trainstep_withgradient!(loss, model, (x, y), opt)
        @test_throws ArgumentError Flux.train!(loss, model, [(x, y)], opt)
    end
end
