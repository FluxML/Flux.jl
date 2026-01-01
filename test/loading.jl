
ls(dims...) = reshape(collect(Float32, 1:prod(dims)), dims...) # accepts dims in reverse order to Dense
dl(nin, nout, bias) = Dense(ls(nout, nin), bias(nout))
dm(bias) = Chain(
  dl(3, 5, bias),
  dl(5, 4, bias),
  dl(4, 3, bias)
)

nobias(n) = false
testdense(m, bt) = @testset "Check layer $i" for (i, (l1, l2)) in enumerate(zip(m, dm(bt)))
  @test l1.weight == l2.weight
  @test l1.bias == l2.bias
  @test_skip typeof(l1.bias) === typeof(l2.bias)
end
  

@testset "loadmodel!(dst, src)" begin
  m1 = Chain(Dense(10 => 5), Dense(5 => 2, relu))
  m2 = Chain(Dense(10 => 5), Dense(5 => 2))
  m3 = Chain(Conv((3, 3), 3 => 16), Dense(5 => 2))
  m4 = Chain(Dense(10 => 6), Dense(6 => 2))
  m5 = Chain(Dense(10 => 5), Parallel(+, Dense(Flux.ones32(2, 5), false), Dense(5 => 2)))
  m6 = Chain(Dense(10 => 5), Parallel(+, Dense(5 => 2), Dense(5 => 2)))

  Flux.loadmodel!(m1, m2)
  # trainable parameters copy over
  @test m1[1].weight == m2[1].weight
  @test m1[1].bias == m2[1].bias
  # non-array leaves are untouched
  @test m1[2].σ == relu

  Flux.loadmodel!(m5, m6)
  # more complex nested structures also work
  @test m5[1].weight == m6[1].weight
  @test m5[2][1].weight == m6[2][1].weight
  # false bias is not overwritten
  @test m5[2][1].bias == false

  # mismatched nodes throw an error
  @test_throws ArgumentError Flux.loadmodel!(m1, m3)
  @test_throws ArgumentError Flux.loadmodel!(m1, m5)
  # size mismatches throw an error
  @test_throws DimensionMismatch Flux.loadmodel!(m1, m4)

  # tests for BatchNorm and Dropout
  m1 = Chain(Conv((3, 3), 3 => 16), BatchNorm(16), Flux.flatten, Dropout(0.2))
  m2 = Chain(Conv((3, 3), 3 => 16), BatchNorm(16), x -> reshape(x, :, size(x)[end]), Dropout(0.1))
  m2[2].μ .= rand(Float32, size(m2[2].μ)...)
  Flux.loadmodel!(m1, m2)
  # non-trainable parameters are copied as well
  @test m1[2].μ == m2[2].μ
  # functions are not copied
  @test m1[3] == Flux.flatten
  # dropout rate is not copied
  @test m1[4].p == 0.2

  # from LegolasFlux (https://github.com/beacon-biosignals/LegolasFlux.jl/blob/80569ab63a8248a8a063c76e0bbf701f4ada9bd4/examples/digits.jl#L33)
  # tests Chain(...) vs Chain([...])
  # tests MaxPool
  # tests testmode!/trainmode! is not copied
  # tests Dense, Conv, BatchNorm, Dropout (like above) but in a bigger model
  chain1 = Chain(Dropout(0.2),
                  Conv((3, 3), 1 => 32, relu),
                  BatchNorm(32, relu),
                  MaxPool((2, 2)),
                  Dropout(0.2),
                  Conv((3, 3), 32 => 16, relu),
                  Dropout(0.2),
                  MaxPool((2, 2)),
                  Dropout(0.2),
                  Conv((3, 3), 16 => 10, relu),
                  Dropout(0.2),
                  x -> reshape(x, :, size(x, 4)),
                  Dropout(0.2),
                  Dense(90 => 10),
                  softmax)
  chain2 = Chain([Dropout(0.1),
                  Conv((3, 3), 1 => 32, relu),
                  BatchNorm(32, relu),
                  MaxPool((3, 3)),
                  Dropout(0.1),
                  Conv((3, 3), 32 => 16, relu),
                  Dropout(0.1),
                  MaxPool((3, 3)),
                  Dropout(0.1),
                  Conv((3, 3), 16 => 10, relu),
                  Dropout(0.1),
                  x -> reshape(x, :, size(x, 4)),
                  Dropout(0.1),
                  Dense(90 => 10),
                  softmax])
  chain2[3].μ .= 5f0
  chain2[3].σ² .= 2f0
  testmode!(chain2)
  Flux.loadmodel!(chain1, chain2)
  for (dst, src) in zip(chain1, chain2)
    if dst isa Dropout
      @test dst.p == 0.2
    elseif dst isa Union{Conv, Dense}
      @test dst.weight == src.weight
      @test dst.bias == src.bias
    elseif dst isa MaxPool
      @test dst.k == (2, 2)
    elseif dst isa BatchNorm
      @test dst.μ == src.μ
      @test dst.σ² == src.σ²
      @test isnothing(dst.active)
    end
  end

  # copy only a subset of the model
  chain1[end - 1].weight .= 1f0
  chain1[3].μ .= 3f0
  chain1[2].bias .= 5f0
  Flux.loadmodel!(chain2[end - 1], chain1[end - 1])
  Flux.loadmodel!(chain2[3], chain1[3])
  @test chain2[end - 1].weight == chain1[end - 1].weight
  @test chain2[3].μ == chain1[3].μ
  @test chain2[2].bias != chain1[2].bias

  # test shared weights
  shared_dst = Dense(10 => 10)
  shared_src = Dense(10 => 10)
  # matched weights are okay
  m1 = Chain(shared_dst, Dense(shared_dst.weight))
  m2 = Chain(shared_src, Dense(shared_src.weight))
  Flux.loadmodel!(m1, m2)
  @test m1[1].weight === m1[2].weight
  @test m1[1].weight == m2[2].weight
  # mismatched weights are an error
  m2 = Chain(Dense(10 => 10), Dense(10 => 10))
  @test_throws ErrorException Flux.loadmodel!(m1, m2)
  # loading into tied weights with absent parameter is okay when the dst == zero
  b = Flux.zeros32(5)
  m1 = Chain(Dense(10 => 5; bias = b), Dense(5 => 5; bias = b))
  m2 = Chain(Dense(10 => 5; bias = Flux.zeros32(5)), Dense(5 => 5; bias = false))
  Flux.loadmodel!(m1, m2)
  @test m1[1].bias === m1[2].bias
  @test iszero(m1[1].bias)
  # loading into tied weights with absent parameter is bad when the dst != zero
  m2[1].bias .= 1
  @test_throws ErrorException Flux.loadmodel!(m1, m2)

  @testset "loadmodel! & filter" begin
    m1 = Chain(Dense(10 => 5), Dense(5 => 2, relu))
    m2 = Chain(Dense(10 => 5), Dropout(0.2), Dense(5 => 2))
    m3 = Chain(Dense(10 => 5), Dense(5 => 2, relu))

    # this will not error cause Dropout is skipped
    Flux.loadmodel!(m1, m2; filter = x -> !(x isa Dropout))
    @test m1[1].weight == m2[1].weight
    @test m1[2].weight == m2[3].weight

    # this will not error cause Dropout is skipped
    Flux.loadmodel!(m2, m3; filter = x -> !(x isa Dropout))
    @test m3[1].weight == m2[1].weight
    @test m3[2].weight == m2[3].weight
  end

  @testset "loadmodel! & absent bias" begin
    m0 = Chain(Dense(2 => 3; bias=false, init = Flux.ones32), Dense(3 => 1))
    m1 = Chain(Dense(2 => 3; bias = Flux.randn32(3)), Dense(3 => 1))
    m2 = Chain(Dense(Float32[1 2; 3 4; 5 6], Float32[7, 8, 9]), Dense(3 => 1))

    Flux.loadmodel!(m1, m2)
    @test m1[1].bias == 7:9
    @test sum(m1[1].weight) == 21

    # load from a model without bias -- should ideally recognise the `false` but `Params` doesn't store it
    m1 = Flux.loadmodel!(m1, m0)
    @test iszero(m1[1].bias)
    @test sum(m1[1].weight) == 6  # written before error

    # load into a model without bias -- should it ignore the parameter which has no home, or error?
    m0 = Flux.loadmodel!(m0, m2)
    @test iszero(m0[1].bias)  # obviously unchanged
    @test sum(m0[1].weight) == 21
  end
end

@testset "loadmodel!(dst, src) with BSON" begin
  m1 = Chain(Dense(Float32[1 2; 3 4; 5 6], Float32[7, 8, 9]), Dense(3 => 1))
  m2 = Chain(Dense(Float32[0 0; 0 0; 0 0], Float32[0, 0, 0]), Dense(3 => 1))
  @test m1[1].weight != m2[1].weight
  mktempdir() do dir
    BSON.@save joinpath(dir, "test.bson") m1
    m2 = Flux.loadmodel!(m2, BSON.load(joinpath(dir, "test.bson"))[:m1])
    @test m1[1].weight == m2[1].weight
  end
end

@testset "state" begin
  m1 = Chain(Dense(10 => 5), Parallel(+, Dense(Flux.ones32(2, 5), false), Dense(5 => 2)))
  m2 = Chain(Dense(10 => 5), Parallel(+, Dense(Flux.zeros32(2, 5), Flux.ones32(2)), Dense(5 => 2)))
  s = Flux.state(m1)
  @test s isa NamedTuple
  @test fieldnames(typeof(s)) == (:layers,)
  @test s.layers isa Tuple
  @test length(s.layers) == 2
  @test s.layers[1].weight === m1[1].weight
  @test s.layers[1].σ === ()
  @test s.layers[2].layers[1].weight === m1[2].layers[1].weight

  Flux.loadmodel!(m2, s)
  @test m2[1].weight == m1[1].weight
  @test all(m2[2].layers[1].bias .== m1[2].layers[1].bias)

  @testset "non-state elements are replaced with empty tuple" begin
    @test Flux.state((1, tanh)) == (1, ())
    @test Flux.state((a=1, b=tanh)) == (; a=1, b=())
    @test Flux.state(Dict(:a=>1, :b=>tanh)) == Dict(:a=>1, :b=>())
    X, Y = Flux.ones32(3, 2), Flux.zeros32(2, 2)
    tree = Dict(:a=>1, :b=>(; c=X, d=(Y, 1, (tanh,)), e=sin))
    state_tree = Dict(:a=>1, :b=>(; c=X, d=(Y, 1, ((),)), e=()))
    @test Flux.state(tree) == state_tree
  end

  @testset "track active state and batch norm params" begin
    m3 = Chain(Dense(10 => 5), Dropout(0.2), Dense(5 => 2), BatchNorm(2))
    trainmode!(m3)
    s = Flux.state(m3)
    @test s.layers[2].active == true
    @test s.layers[2].p == 0.2
    @test s.layers[4].λ === ()
    for k in (:β, :γ, :μ, :σ², :ϵ, :momentum, :affine, :track_stats, :active, :chs)
      @test s.layers[4][k] === getfield(m3[4], k)
    end
  end

  @testset "preservation of saved types" begin
    m = (num = 1, cnum = Complex(1.2, 2), str = "hello", arr = [1, 2, 3], 
        bool = true, dict = Dict(:a => 1, :b => 2), tup = (1, 2, 3), 
        sym = :a, nth = nothing)

    s = Flux.state(m)
    @test s == m
  end
end
