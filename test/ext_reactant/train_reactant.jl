@testset "Flux.train! on Reactant device" begin
    dev = MLDataDevices.reactant_device(force=true)
    cpu = cpu_device()

    model0 = Chain(Dense(4 => 8, tanh), Dense(8 => 2))
    X = randn(Float32, 4, 16)
    Y = randn(Float32, 2, 16)

    # Move the model to the device *before* `setup`, so the optimiser rule is auto-wrapped for
    # Reactant (keeps Adam's βt as an on-device tracked number rather than a frozen constant).
    model = model0 |> dev
    x, y = X |> dev, Y |> dev
    opt = Flux.setup(Adam(1f-2), model)

    loss(m, a, b) = Flux.mse(m(a), b)

    l0 = Reactant.to_number(Reactant.@jit loss(model, x, y))
    w0 = model.layers[1].weight |> cpu

    # Fixed batch shape → the step is compiled once and reused across all iterations.
    data = [(x, y) for _ in 1:30]
    Flux.train!(loss, model, data, opt)

    l1 = Reactant.to_number(Reactant.@jit loss(model, x, y))
    w1 = model.layers[1].weight |> cpu

    @test l1 < l0            # loss decreased
    @test !(w0 ≈ w1)         # parameters were updated in place

    # A smaller final batch forces a second compile — the shape-keyed cache must handle it.
    x2, y2 = (X[:, 1:8] |> dev), (Y[:, 1:8] |> dev)
    data2 = [(x, y), (x, y), (x2, y2)]   # two shapes in one call
    @test Flux.train!(loss, model, data2, opt) === nothing

    # Host-resident data must be rejected on the Reactant path.
    @test_throws ArgumentError Flux.train!(loss, model, [(X, Y)], opt)
end

@testset "Flux.train_step! on Reactant device" begin
    dev = MLDataDevices.reactant_device(force=true)
    cpu = cpu_device()

    model0 = Chain(Dense(4 => 8, tanh), Dense(8 => 2))
    X = randn(Float32, 4, 16)
    Y = randn(Float32, 2, 16)

    model = model0 |> dev
    x, y = X |> dev, Y |> dev
    opt = Flux.setup(Adam(1f-2), model)

    loss(m, a, b) = Flux.mse(m(a), b)

    l0 = Reactant.to_number(Reactant.@jit loss(model, x, y))
    w0 = model.layers[1].weight |> cpu

    # `train_step!` returns just the host-scalar loss, mutating model/opt in place.
    l = Flux.train_step!(loss, model, (x, y), opt)
    @test l isa Real && isfinite(l)
    @test l ≈ l0                               # measured before the update
    @test !(w0 ≈ model.layers[1].weight |> cpu)            # parameters updated in place

    # `train_step_withgradient!` also returns the on-device gradient.
    l2, g = Flux.train_step_withgradient!(loss, model, (x, y), opt)
    @test l2 isa Real && isfinite(l2)
    @test Flux.get_device_type(g) <: Flux.ReactantDevice   # gradient stays on the device

    # Looping train_step! keeps reducing the loss (reuses the cached executable).
    for _ in 1:30
        Flux.train_step!(loss, model, (x, y), opt)
    end
    @test Reactant.to_number(Reactant.@jit loss(model, x, y)) < l0

    # Host-resident data must be rejected by both entry points.
    @test_throws ArgumentError Flux.train_step!(loss, model, (X, Y), opt)
    @test_throws ArgumentError Flux.train_step_withgradient!(loss, model, (X, Y), opt)
end

@testset "Flux.train_step! with a structured (non-array) batch element" begin
    dev = MLDataDevices.reactant_device(force=true)

    model = Dense(4 => 2) |> dev
    # The batch element is a NamedTuple of arrays, not a bare array — the compile-cache key must
    # summarise its shape by recursing through the leaves rather than calling `size` on it directly.
    nt = (x = randn(Float32, 4, 8), y = randn(Float32, 2, 8)) |> dev
    opt = Flux.setup(Adam(1f-2), model)
    lossnt(m, b) = Flux.mse(m(b.x), b.y)

    l0 = Reactant.to_number(Reactant.@jit lossnt(model, nt))
    # Exercise both compiled variants (loss-only and with-gradient) with the structured batch.
    l, g = Flux.train_step_withgradient!(lossnt, model, (nt,), opt)
    @test l isa Real && isfinite(l)
    @test Flux.get_device_type(g) <: Flux.ReactantDevice
    for _ in 1:20
        Flux.train_step!(lossnt, model, (nt,), opt)
    end
    @test Reactant.to_number(Reactant.@jit lossnt(model, nt)) < l0
end

@testset "auxiliary loss outputs on Reactant" begin
    dev = MLDataDevices.reactant_device(force=true)

    model = Chain(Dense(4 => 8, tanh), Dense(8 => 2)) |> dev
    x, y = randn(Float32, 4, 16) |> dev, randn(Float32, 2, 16) |> dev
    opt = Flux.setup(Adam(1f-2), model)

    scalarloss(m, a, b) = Flux.mse(m(a), b)
    # loss returns (scalar loss, NamedTuple of a device-computed statistic)
    auxloss(m, a, b) = (Flux.mse(m(a), b), (; sumsq = sum(abs2, m(a))))

    l0 = Reactant.to_number(Reactant.@jit scalarloss(model, x, y))

    v = Flux.train_step!(auxloss, model, (x, y), opt)
    @test v isa Tuple                                              # full value returned
    @test v[1] isa Real && isfinite(v[1])                         # scalar loss read to host
    @test v[1] ≈ l0                                               # measured before the update
    @test Flux.get_device_type(v[2].sumsq) <: Flux.ReactantDevice # aux stays on the device

    # with-gradient variant: same value shape, gradient stays on the device
    v2, g = Flux.train_step_withgradient!(auxloss, model, (x, y), opt)
    @test v2 isa Tuple && v2[1] isa Real && isfinite(v2[1])
    @test Flux.get_device_type(g) <: Flux.ReactantDevice

    # differentiating `first∘loss`, training with the aux loss still reduces the loss
    for _ in 1:20
        Flux.train_step!(auxloss, model, (x, y), opt)
    end
    @test Reactant.to_number(Reactant.@jit scalarloss(model, x, y)) < l0

    # `train!` over an aux-returning loss runs (it guards on the scalar and discards the aux)
    @test Flux.train!(auxloss, model, [(x, y) for _ in 1:5], opt) === nothing
end

@testset "Reactant compile-cache auto-eviction" begin
    dev = MLDataDevices.reactant_device(force=true)
    ext = Base.get_extension(Flux, :FluxReactantExt)
    @test ext !== nothing
    loss(m, a, b) = Flux.mse(m(a), b)

    empty!(ext.COMPILE_CACHE)   # clean slate for deterministic counts

    # Compile a step for a fresh model and return only the model, so the batch/optimiser locals are
    # function-scoped and don't leak into this testset (which would keep them, and the model, alive).
    compile_one() = let model = Chain(Dense(4 => 8, tanh), Dense(8 => 2)) |> dev
        x, y = randn(Float32, 4, 16) |> dev, randn(Float32, 2, 16) |> dev
        opt = Flux.setup(Adam(1f-2), model)
        Flux.train_step!(loss, model, (x, y), opt)
        model
    end

    models = Any[compile_one() for _ in 1:3]
    @test length(ext.COMPILE_CACHE) == 3   # one entry per distinct model

    # Once the models are unreachable and collected, their entries (and executables) are evicted. We
    # assert `<= 1` rather than `== 0` only because Julia's GC is conservative about the most recently
    # created object, whose reference can linger on the stack; the point is that the dropped models'
    # entries are freed, not retained for the session.
    empty!(models)
    GC.gc(true); GC.gc(true)
    ext._prune_compile_cache!()
    @test length(ext.COMPILE_CACHE) <= 1

    # A live model keeps *its* entry across pruning (and keeps training).
    model = Chain(Dense(4 => 8, tanh), Dense(8 => 2)) |> dev
    x, y = randn(Float32, 4, 16) |> dev, randn(Float32, 2, 16) |> dev
    opt = Flux.setup(Adam(1f-2), model)
    l0 = Reactant.to_number(Reactant.@jit loss(model, x, y))
    for _ in 1:20
        Flux.train_step!(loss, model, (x, y), opt)
    end
    l1 = Reactant.to_number(Reactant.@jit loss(model, x, y))
    GC.gc(true)
    ext._prune_compile_cache!()
    handle = first(Flux.trainables(model))
    @test any(v -> v[1].value === handle, values(ext.COMPILE_CACHE))   # this model's entry survived
    @test l1 < l0
end
