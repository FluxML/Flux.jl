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
            @test l0 isa T
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
