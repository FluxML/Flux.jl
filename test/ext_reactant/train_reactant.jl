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
