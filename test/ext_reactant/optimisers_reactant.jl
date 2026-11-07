# `trainstep!` on a Reactant device follows the eager trajectory. Every rule costs a compilation, so
# these are picked for their distinct code paths: step counters, `onevalue`, sign and `ifelse`, chains.
@testset "Reactant optimiser trajectories match eager" begin
    rules = [
        Descent(0.1f0), Momentum(0.05f0), RMSProp(0.01f0), AdaGrad(0.05f0), Rprop(0.01f0), Lion(0.01f0),
        Adam(0.01f0), AdamW(0.01f0, (0.9f0, 0.999f0), 0.1f0), AMSGrad(0.01f0), RAdam(0.01f0),
        OptimiserChain(ClipGrad(0.05f0), ClipNorm(0.5f0), WeightDecay(0.1f0), Adam(0.01f0)),
        OptimiserChain(AccumGrad(3), Adam(0.01f0)),
        OptimiserChain(AccumGrad(2), ClipNorm(0.5f0), AdamW(0.01f0, (0.9f0, 0.999f0), 0.1f0)),
        MixedPrecision(Float64, OptimiserChain(AccumGrad(3), Adam(0.01f0))),
    ]
    for rule in rules
        @testset "$rule" begin
            @test reactant_trajectory_error(rule) < 1e-5
        end
    end
end

# The compiled step only accumulates while every rule starts with `AccumGrad` and none completes:
# check the cases where the counters drift apart (a frozen layer) and where a rule doesn't accumulate.
@testset "Reactant AccumGrad with frozen layers and mixed rules" begin
    dev = MLDataDevices.reactant_device(force = true)
    loss(m, x, y) = Flux.mse(m(x), y)
    model0 = Chain(Dense(4 => 8, tanh), Dense(8 => 2))
    batches = [(randn(Float32, 4, 16), randn(Float32, 2, 16)) for _ in 1:9]
    accum = OptimiserChain(AccumGrad(3), Adam(0.01f0))

    # The first layer is frozen for the first two steps, so it completes its accumulation two steps
    # after the second layer.
    frozen(m) = Flux.setup(accum, m)
    function frozen_hook(opt, k)
        k == 1 && Flux.freeze!(opt.layers[1])
        k == 3 && Flux.thaw!(opt.layers[1])
    end
    # Only the first layer accumulates its gradients.
    mixed(m) = (; layers = (Flux.setup(accum, m).layers[1], Flux.setup(Adam(0.01f0), m.layers[2])))

    @testset "$name" for (name, setup, hook) in (("frozen layer", frozen, frozen_hook),
                                                 ("mixed rules", mixed, (opt, k) -> nothing))
        eager, re = deepcopy(model0), deepcopy(model0) |> dev
        opt_eager, opt_re = setup(eager), setup(re)
        reactant_full_precision() do
            for (k, b) in enumerate(batches)
                hook(opt_eager, k)
                hook(opt_re, k)
                l_eager = Flux.trainstep!(loss, eager, b, opt_eager)
                l_re = Flux.trainstep!(loss, re, b |> dev, opt_re)
                @test l_re ≈ l_eager rtol=1e-5
            end
        end
        for (p, q) in zip(Flux.trainables(re |> cpu_device()), Flux.trainables(eager))
            @test p ≈ q rtol=1e-5
        end
    end
end

@testset "Reactant trainstep! in low precision" begin
    dev = MLDataDevices.reactant_device(force = true)
    loss(m, x, y) = Flux.mse(m(x), y)
    X, Y = randn(Float32, 4, 16), randn(Float32, 2, 16)

    for (prec, T) in ((f16, Float16), (bf16, BFloat16))
        @testset "$T" begin
            model = Chain(Dense(4 => 8, tanh), Dense(8 => 2))
            l32 = loss(model, X, Y)
            model = model |> prec |> dev
            x, y = (X, Y) |> prec |> dev
            # Not Adam, which diverges here eagerly too: its default ϵ = 1e-8 underflows in Float16,
            # and β₂ = 0.999 rounds to 1 in BFloat16.
            opt = Flux.setup(Descent(0.1f0), model)
            l0 = Flux.trainstep!(loss, model, (x, y), opt)
            @test l0 isa Float32  # losses upcast low-precision inputs
            @test l0 ≈ l32 rtol=2e-2
            l = l0
            for _ in 1:20
                l = Flux.trainstep!(loss, model, (x, y), opt)
            end
            @test isfinite(l) && l < l0
            @test all(p -> eltype(p) == T, Flux.trainables(model))
        end
    end
end

@testset "Reactant keeps Float64" begin
    dev = MLDataDevices.reactant_device(force = true)
    loss(m, x, y) = Flux.mse(m(x), y)
    X, Y = randn(4, 16), randn(2, 16)

    eager = f64(Chain(Dense(4 => 8, tanh), Dense(8 => 2)))
    model = eager |> dev
    @test all(p -> eltype(p) == Float64, Flux.trainables(model))
    x, y = (X, Y) |> dev
    @test eltype(x) == Float64

    opt_eager, opt = Flux.setup(Descent(0.1), eager), Flux.setup(Descent(0.1), model)
    for _ in 1:3
        l_eager = Flux.trainstep!(loss, eager, (X, Y), opt_eager)
        l = Flux.trainstep!(loss, model, (x, y), opt)
        @test l isa Float64
        # far below Float32 resolution: the step really runs in Float64
        @test l ≈ l_eager rtol=1e-12
    end
    @test all(p -> eltype(p) == Float64, Flux.trainables(model))
    for (p, q) in zip(Flux.trainables(model |> cpu_device()), Flux.trainables(eager))
        @test p ≈ q rtol=1e-12
    end
end
