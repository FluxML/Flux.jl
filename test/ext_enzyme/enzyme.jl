using Enzyme: Enzyme, Duplicated, Const, Active

@testset "Models" begin
    function loss(model, x)
        mean(model(x))
    end

    models_xs = [
        (Dense(2=>4), randn(Float32, 2), "Dense"),
        (Chain(Dense(2=>4, tanh), Dense(4=>3)), randn(Float32, 2), "Chain(Dense, Dense)"),
        (f64(Chain(Dense(2=>4), Dense(4=>2))), randn(Float64, 2, 1), "f64(Chain(Dense, Dense))"),
        (Flux.Scale([1.0f0 2.0f0 3.0f0 4.0f0], true, abs2), randn(Float32, 2), "Flux.Scale"),
        (Conv((3, 3), 2 => 3), randn(Float32, 3, 3, 2, 1), "Conv"),
        (Chain(Conv((3, 3), 2 => 3, ), Conv((3, 3), 3 => 1, tanh)), rand(Float32, 5, 5, 2, 1), "Chain(Conv, Conv)"),
        (Chain(Conv((4, 4), 2 => 2, pad=SamePad()), MeanPool((5, 5), pad=SamePad())), rand(Float32, 5, 5, 2, 2), "Chain(Conv, MeanPool)"),
        (Maxout(() -> Dense(5 => 4, tanh), 3), randn(Float32, 5, 1), "Maxout"),
        (SkipConnection(Dense(2 => 2), vcat), randn(Float32, 2, 3), "SkipConnection"),
        (Flux.Bilinear((2, 2) => 3), randn(Float32, 2, 1), "Bilinear"),
        (ConvTranspose((3, 3), 3 => 2, stride=2), rand(Float32, 5, 5, 3, 1), "ConvTranspose"),
        # (first ∘ LayerNorm(2), randn(Float32, 2, 10), "LayerNorm"),
        # (BatchNorm(2), randn(Float32, 2, 10), "BatchNorm"),
        (first ∘ MultiHeadAttention(16), randn32(16, 20, 2), "MultiHeadAttention"),
    ]

    for (model, x, name) in models_xs
        @testset "Enzyme grad check $name" begin
            println("testing $name with Enzyme")
            test_gradients(model, x; loss, compare_finite_diff=false, test_enzyme=true)
        end
    end
end

@testset "Recurrent Layers" begin
    function loss(model, x)
        mean(model(x))
    end

    models_xs = [
        # (RNN(3 => 2), randn(Float32, 3, 2), "RNN"), 
        # (LSTM(3 => 5), randn(Float32, 3, 2), "LSTM"),
        # (GRU(3 => 5), randn(Float32, 3, 10), "GRU"),
        # (Chain(RNN(3 => 4), RNN(4 => 3)), randn(Float32, 3, 2), "Chain(RNN, RNN)"),
        # (Chain(LSTM(3 => 5), LSTM(5 => 3)), randn(Float32, 3, 2), "Chain(LSTM, LSTM)"),
    ]

    for (model, x, name) in models_xs
        @testset "check grad $name" begin
            println("testing $name")
            test_gradients(model, x; loss, compare_finite_diff=false, test_enzyme=true)
        end
    end
end

@testset "gradient, withgradient, Duplicated" begin
    # Tests above are about how Enzyme digests Flux layers.
    # Tests here are just the interface Flux.gradient(f, Duplicated(model)) etc.
    m1 = Duplicated(Dense(3=>2))
    @test m1 isa Duplicated
    g1 = Flux.gradient(m -> sum(m.bias), m1) |> only
    @test iszero(g1.weight)
    @test g1.bias == [1, 1]
    @test m1.dval.bias == [1, 1]

    g2 = Flux.withgradient((m,x) -> sum(m(x)), m1, [1,2,3f0])
    @test g2.val ≈ sum(m1([1,2,3f0]))
    @test g2.grad[1].weight ≈ [1 2 3; 1 2 3]
    @test g2.grad[2] === nothing  # implicitly Const

    g3 = Flux.withgradient(Duplicated([1,2,4.], zeros(3))) do x
              z = 1 ./ x
              sum(z), z  # here z is an auxillary output
           end
    @test g3.grad[1] ≈ [-1.0, -0.25, -0.0625]
    @test g3.val[1] ≈ 1.75
    @test g3.val[2] ≈ [1.0, 0.5, 0.25]
    g4 = Flux.withgradient(Duplicated([1,2,4.], zeros(3))) do x
              z = 1 ./ x
              (loss=sum(z), aux=string(z))
           end
    @test g4.grad[1] ≈ [-1.0, -0.25, -0.0625]
    @test g4.val.loss ≈ 1.75
    @test g4.val.aux == "[1.0, 0.5, 0.25]"

    # setup understands Duplicated:
    @test Flux.setup(Adam(), m1) == Flux.setup(Adam(), m1.val)

    # state, loadmodel do too -- all ignore the dval branch, no outer (; val, dval) namedtuple
    @test Flux.state(m1) == Flux.state(m1.val)
    oldmodel = deepcopy(m1)
    oldpar = deepcopy(Flux.state(m1))
    m1.val.weight .= 0
    @test Flux.loadmodel!(m1, oldmodel).val.weight ≈ oldpar.weight
    m1.val.weight .= 0
    @test Flux.loadmodel!(m1, oldpar).val.weight ≈ oldpar.weight

    # At least one Duplicated is required:
    @test_throws ArgumentError Flux.gradient(m -> sum(m.bias), Const(m1.val))
    @test_throws ArgumentError Flux.gradient((m,x) -> sum(m(x)), Const(m1.val), [1,2,3f0])
    @test_throws ArgumentError Flux.withgradient(m -> sum(m.bias), Const(m1.val))
    @test_throws ArgumentError Flux.withgradient((m,x) -> sum(m(x)), Const(m1.val), [1,2,3f0])
    # Active is disallowed:
    @test_throws ArgumentError Flux.gradient((m,z) -> sum(m.bias)/z, m1, Active(3f0))
    @test_throws ArgumentError Flux.gradient((m,z) -> sum(m.bias)/z, m1.val, Active(3f0))
    @test_throws ArgumentError Flux.gradient((m,z) -> sum(m.bias)/z, Const(m1.val), Active(3f0))
    # Duplicated
    @test_throws Exception Flux.gradient((m,z) -> sum(m.bias)/z, m1, Duplicated(3f0, 0f0))

    # Using Duplicated within Zygote.gradient is not supported:
    @test_throws ErrorException Zygote.gradient((m,x) -> sum(m(x)), m1, [1,2,3f0])
end

@testset "bugs found" begin
    _duplicated(x) = Duplicated(x, Enzyme.make_zero(x))
    z = _duplicated(zeros32(3))
    @test_broken Flux.gradient(sum ∘ LayerNorm(3), z)[1] ≈ [0.0, 0.0, 0.0]  # Constant memory is stored (or returned) to a differentiable variable
    @test Flux.gradient(|>, z, _duplicated(sum ∘ LayerNorm(3)))[1] ≈ [0.0, 0.0, 0.0]
    @test Flux.gradient(|>, z, Const(sum ∘ LayerNorm(3)))[2] === nothing broken=VERSION >= v"1.11"

    @test_broken Flux.withgradient(sum ∘ LayerNorm(3), z).grad[1] ≈ [0.0, 0.0, 0.0]  # AssertionError: Base.allocatedinline(actualRetType) returns false: actualRetType = Any, rettype = Active{Any}
    @test_broken Flux.withgradient(|>, z, _duplicated(sum ∘ LayerNorm(3))).grad[1] ≈ [0.0, 0.0, 0.0]
end
