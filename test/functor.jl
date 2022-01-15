using Flux: loadparams!, Zeros, destructure
    
function build_test_chain(fbias)
  ls(dims...) = reshape(collect(Float32, 1:prod(dims)), reverse(dims)...)
  
  Chain(
      Dense(ls(3, 5), fbias(5)),
      Dense(ls(5, 4), fbias(4)),
      Dense(ls(4, 3), fbias(3))
      )
end

nobias(n) = Zeros()

function test_chains_equal(m1, m2)
  @testset "Check layer $i" for (i, (l1, l2)) in enumerate(zip(m1, m2))
    @test l1.weight == l2.weight
    @test l1.bias == l2.bias
    @test typeof(l1.bias) === typeof(l2.bias)
  end
end

@testset "Params" begin
  m = Dense(10, 5)
  @test size.(params(m)) == [(5, 10), (5,)]
  m = RNN(10, 5)
  @test size.(params(m)) == [(5, 10), (5, 5), (5,), (5, 1)]
  
  # Layer duplicated in same chain, params just once pls.
  c = Chain(m, m)
  @test size.(params(c)) == [(5, 10), (5, 5), (5,), (5, 1)]
  
  # Self-referential array. Just want params, no stack overflow pls.
  r = Any[nothing,m]
  r[1] = r
  @test size.(params(r)) == [(5, 10), (5, 5), (5,), (5, 1)]
  
  @testset "use params in gradient context" begin
    m = Chain(Dense(3,2), Dense(2,2))
    ps = Flux.params(m)
    gs = gradient(() -> sum(sum(p) for p in Flux.params(m)), ps)
    for p in ps
      @test gs[p] ≈ ones(size(p))
    end    
    
    w1, w2 =  rand(2), rand(2)
    ps = Flux.params(w1, w2)
    gs = gradient(() -> sum(sum(p) for p in Flux.params(w1, w2)), ps)
    for p in ps
      @test gs[p] ≈ ones(size(p))
    end
    
    m = Chain(Dense(3,2), Dense(2,2))
    g = gradient(m -> sum(params(m)[1]), m)[1]
    @test g.layers[1].weight == ones(Float32, 2, 3)
    
    gs = gradient(() -> sum(params(m)[1]), params(m))
    @test gs[params(m)[1]] == ones(Float32, 2, 3)
    
    # Tests from https://github.com/FluxML/Flux.jl/pull/1614
    m = Dense(3, 2)
    ps = Flux.params(m)
    data = rand(Float32, 3, 5)
    loss(m, x) = sum(m(x).^2)
    
    g1 = gradient(Flux.params(m)) do
      loss(m, data)
    end
    g2 = gradient(Flux.params(m)) do
      ps = Flux.params(m) # just creating params without using them
      loss(m, data)
    end
    g3 = gradient(Flux.params(m)) do
      ps = Flux.params(m)
      loss(m, data) + sum(sum(p) for p in ps)
    end 
    g4 = gradient(Flux.params(m)) do
      loss(m, data) + sum(sum(p) for p in ps)
    end
    g5 = gradient(Flux.params(m)) do
      sum(Flux.params(m)[1]) + sum(Flux.params(m)[2])
    end
    g6 = gradient(Flux.params(m)) do
      sum(ps[1]) + sum(ps[2])
    end
    @test g2[m.weight] == g1[m.weight]
    @test g3[m.weight] == g1[m.weight] .+ 1
    @test g4[m.weight] == g1[m.weight] .+ 1
    @test all(g5[m.weight] .== 1)
    @test_broken all(g6[m.weight] .== 1)
  end
end


@testset "Param remapping" begin
  @testset "loadparams!" begin
    pars(w, b) = [w, b]
    
    pars(w, b::Zeros) = [w, Flux.zeros32(size(w,1))]
    pars(l) = pars(l.weight, l.bias)
    pararray(m) = mapreduce(pars, vcat, m)
    weights(m) = mapreduce(l -> [l.weight], vcat, m)
    @testset "Bias type $bt" for bt in (Flux.zeros32, nobias)
      m = build_test_chain(bt)
      loadparams!(m, params(m))
      test_chains_equal(m, build_test_chain(bt))
    end
    
    @testset "$b1 to $b2" for (b1, b2, be) in (
      (Flux.zeros32, Flux.ones32, Flux.ones32),   # Load ones as bias to a model with zeros as bias -> model gets ones as bias
      (Flux.ones32, nobias, Flux.zeros32), # Load Zeros as bias to a model with ones as bias-> model gets zeros as bias
      (nobias, Flux.ones32, nobias),     # Load ones as bias to a model with Zeros as bias-> model bias does not change
      )
      m1 = build_test_chain(b1)
      m2 = build_test_chain(b2)
      loadparams!(m1, b1 == nobias ? weights(m2) : pararray(m2))
      test_chains_equal(m1, build_test_chain(be))
    end
  end
end

@testset "Destructure" begin
  @testset "Bias type $bt" for bt in (zeros, nobias)
    m = build_test_chain(bt)
    p, re = destructure(m)
    test_chains_equal(re(p), build_test_chain(bt))
  end
  
  @testset "restructure in gradient" begin
    x = rand(Float32, 3, 1)
    m = build_test_chain(zeros)
    ∇m = gradient(m -> sum(m(x)), m)[1]
    p, re = destructure(m)
    ∇p = gradient(θ -> sum(re(θ)(x)), p)[1]
    @test ∇p ≈ destructure(∇m)[1] rtol=1e-6
  end
  
  @testset "destructure with buffers" begin
    p, re = destructure(BatchNorm(3))
    @test length(p) == 6
    
    # https://github.com/FluxML/Flux.jl/issues/1727
    x = rand(Float32, 3, 4)
    y, back = Flux.pullback(x, p) do x, p
      vec(re(p)(x))
    end
    @test_nowarn back(y)
    b = back(y)
    @test size(b[1]) == size(x)
    @test size(b[2]) == size(p)
  end
end

@testset "Train and test mode" begin
  mutable struct DummyLayer
    testing::Bool
  end
  Flux.testmode!(m::DummyLayer, testing=true) = (m.testing = testing; m)
  
  c = Chain(DummyLayer(true))
  testmode!(c)
  @test c[1].testing
  trainmode!(c)
  @test !c[1].testing
end
