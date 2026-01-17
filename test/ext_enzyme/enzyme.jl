# ENZYME CPU TESTS

@testset "enzyme gradients" begin
    for (model, x, name) in TEST_MODELS
        @testset "Enzyme grad check $name" begin
            @test test_gradients(model, x; reference=AutoZygote(), compare=AutoEnzyme())
        end
    end
end

@testset "gradient, withgradient, Duplicated" begin
    # Tests above are about how Enzyme digests Flux layers.
    # Tests here are just the interface Flux.gradient(f, Duplicated(model)) etc.
    m1 = Duplicated(Dense(3=>2))
    @test m1 isa Enzyme.Duplicated
    g1 = Flux.gradient(m -> sum(m.bias), m1) |> only
    @test iszero(g1.weight)
    @test g1.bias == [1, 1]
    @test m1.dval.bias == [1, 1]

    g2 = Flux.withgradient((m,x) -> sum(m(x)), m1, Const([1,2,3f0]))
    @test g2.val ≈ sum(m1([1,2,3f0]))
    @test g2.grad[1].weight ≈ [1 2 3; 1 2 3]
    @test g2.grad[2] === nothing

    ## Auxillary outputs not supported at the moment
    # g3 = Flux.withgradient(Duplicated([1,2,4.], zeros(3))) do x
    #           z = 1 ./ x
    #           sum(z), z  # here z is an auxillary output
    #        end
    # @test g3.grad[1] ≈ [-1.0, -0.25, -0.0625]
    # @test g3.val[1] ≈ 1.75
    # @test g3.val[2] ≈ [1.0, 0.5, 0.25]
    # g4 = Flux.withgradient(Duplicated([1,2,4.], zeros(3))) do x
    #           z = 1 ./ x
    #           (loss=sum(z), aux=string(z))
    #        end
    # @test g4.grad[1] ≈ [-1.0, -0.25, -0.0625]
    # @test g4.val.loss ≈ 1.75
    # @test g4.val.aux == "[1.0, 0.5, 0.25]"

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

    # Only Const args are supported
    @test Flux.gradient(m -> sum(m.bias), Const(m1.val))[1] === nothing
    @test Flux.gradient((m,x) -> sum(m(x)), Const(m1.val), [1,2,3f0]) isa Tuple{Nothing,Vector{Float32}}
    @test Flux.withgradient(m -> sum(m.bias), Const(m1.val)).grad[1] === nothing
    @test Flux.withgradient((m,x) -> sum(m(x)), Const(m1.val), [1,2,3f0]).grad isa Tuple{Nothing,Vector{Float32}}
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
    @test Flux.gradient(sum ∘ LayerNorm(3), z)[1] ≈ [0.0, 0.0, 0.0]
    @test Flux.gradient(|>, z, _duplicated(sum ∘ LayerNorm(3)))[1] ≈ [0.0, 0.0, 0.0]
    @test Flux.gradient(|>, z, Const(sum ∘ LayerNorm(3)))[2] === nothing
    @test Flux.withgradient(sum ∘ LayerNorm(3), z).grad[1] ≈ [0.0, 0.0, 0.0]
    @test Flux.withgradient(|>, z, _duplicated(sum ∘ LayerNorm(3))).grad[1] ≈ [0.0, 0.0, 0.0]
end
