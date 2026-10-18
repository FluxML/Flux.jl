@testset "non-Enzyme AD backend rejected on Reactant" begin
    dev = MLDataDevices.reactant_device(force=true)
    ext = Base.get_extension(Flux, :FluxReactantExt)
    loss(m, a, b) = Flux.mse(m(a), b)

    model = Chain(Dense(4 => 8, tanh), Dense(8 => 2)) |> dev
    x, y = randn(Float32, 4, 16) |> dev, randn(Float32, 2, 16) |> dev
    opt = Flux.setup(Adam(1f-2), model)
    w0 = Array(model[1].weight)
    ncache = length(ext.COMPILE_CACHE)

    # Rejected up front, before anything is traced or compiled.
    for adtype in (AutoZygote(), AutoMooncake(), AutoFiniteDifferences(; fdm = _default_fdm()))
        @test_throws ArgumentError Flux.trainstep!(loss, adtype, model, (x, y), opt)
        @test_throws ArgumentError Flux.trainstep_withgradient!(loss, adtype, model, (x, y), opt)
        @test_throws ArgumentError Flux.train!(loss, adtype, model, [(x, y)], opt)
    end
    err = try Flux.trainstep!(loss, AutoZygote(), model, (x, y), opt) catch e; e end
    @test occursin("Enzyme", err.msg)
    @test length(ext.COMPILE_CACHE) == ncache
    @test Array(model[1].weight) == w0

    @test Flux.trainstep!(loss, AutoEnzyme(), model, (x, y), opt) isa Real
end

# Host copies of every array and number in `x`, e.g. a model with its optimiser state.
_host_leaves(x) = [l isa Number ? Reactant.to_number(l) : Array(l)
                   for l in Functors.fleaves(x) if l isa Union{Number, AbstractArray{<:Number}}]

@testset "non-finite loss skips the compiled update on Reactant" begin
    dev = MLDataDevices.reactant_device(force=true)
    loss(m, a, b) = Flux.mse(m(a), b)
    auxloss(m, a, b) = (loss(m, a, b), (; s = sum(m(a))))

    model0 = Chain(Dense(4 => 8, tanh), Dense(8 => 2))
    X, Y = randn(Float32, 4, 16), randn(Float32, 2, 16)
    x, y = X |> dev, Y |> dev
    # Same shapes as `(x, y)`, so these reuse the executables compiled for it.
    bad = [(fill(NaN32, 4, 16) |> dev, y), (x, fill(Inf32, 2, 16) |> dev)]

    model = deepcopy(model0) |> dev
    opt = Flux.setup(Adam(1f-2), model)
    ref = deepcopy(model0) |> dev   # trained on the finite batches only
    opt_ref = Flux.setup(Adam(1f-2), ref)

    Flux.trainstep!(loss, model, (x, y), opt)   # non-zero Adam moments
    Flux.trainstep!(loss, ref, (x, y), opt_ref)
    before = _host_leaves((model, opt))
    @test before == _host_leaves((ref, opt_ref))

    for b in bad
        @test !isfinite(Flux.trainstep!(loss, model, b, opt))
        @test _host_leaves((model, opt)) == before

        l, g = Flux.trainstep_withgradient!(loss, model, b, opt)
        @test !isfinite(l)
        @test Flux.get_device_type(g) <: Flux.ReactantDevice
        @test _host_leaves((model, opt)) == before

        v = Flux.trainstep!(auxloss, model, b, opt)
        @test !isfinite(v[1])
        @test _host_leaves((model, opt)) == before

        @test_throws DomainError Flux.train!(loss, model, [b, (x, y)], opt)
        @test _host_leaves((model, opt)) == before
    end

    # Training then carries on exactly as if the bad batches had never been seen.
    for _ in 1:3
        l = Flux.trainstep!(loss, model, (x, y), opt)
        l_ref = Flux.trainstep!(loss, ref, (x, y), opt_ref)
        @test isfinite(l) && l ≈ l_ref
    end
    Flux.train!(loss, model, [(x, y)], opt)
    Flux.train!(loss, ref, [(x, y)], opt_ref)
    for (a, b) in zip(_host_leaves((model, opt)), _host_leaves((ref, opt_ref)))
        @test a ≈ b
    end
    @test Reactant.to_number(Reactant.@jit loss(model, x, y)) < Flux.mse(model0(X), Y)
end
