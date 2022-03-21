using Flux
using Flux: throttle, nfan, glorot_uniform, glorot_normal,
             kaiming_normal, kaiming_uniform, orthogonal, truncated_normal,
             sparse_init, identity_init, stack, unstack, batch, unbatch,
             unsqueeze, params, loadparams!
using StatsBase: var, std
using Statistics, LinearAlgebra
using Random
using Test

@testset "Throttle" begin
  @testset "default behaviour" begin
    a = []
    f = throttle(()->push!(a, time()), 1, leading=true, trailing=false)
    f()
    f()
    f()
    sleep(1.01)
    @test length(a) == 1
  end

  @testset "leading behaviour" begin
    a = []
    f = throttle(()->push!(a, time()), 1, leading=true, trailing=false)
    f()
    @test length(a) == 1
    f()
    @test length(a) == 1
    sleep(1.01)
    f()
    @test length(a) == 2
  end

  @testset "trailing behaviour" begin
    a = []
    f = throttle(()->push!(a, time()), 1, leading=false, trailing=true)
    f()
    @test length(a) == 0
    f()
    @test length(a) == 0
    sleep(1.01)
    @test length(a) == 1
  end

  @testset "arguments" begin
    a = []
    f = throttle((x)->push!(a, x), 1, leading=true, trailing=true)
    f(1)
    @test a == [1]
    f(2)
    @test a == [1]
    f(3)
    @test a == [1]
    sleep(1.01)
    @test a == [1, 3]
  end
end

@testset "Initialization" begin
  # Set random seed so that these tests don't fail randomly
  Random.seed!(0)

  @testset "Fan in/out" begin
    @test nfan() == (1, 1) #For a constant
    @test nfan(100) == (1, 100) #For vector
    @test nfan(100, 200) == (200, 100) == nfan((100, 200)) #For Dense layer
    @test nfan(2, 30, 40) == (2 * 30, 2 * 40) #For 1D Conv layer
    @test nfan(2, 3, 40, 50) == (2 * 3 * 40, 2 * 3 * 50) #For 2D Conv layer
    @test nfan(2, 3, 4, 50, 60) == (2 * 3 * 4 * 50, 2 * 3 * 4 * 60) #For 3D Conv layer
  end

  @testset "Basics: $init" for init in [
      glorot_uniform, glorot_normal, 
      kaiming_uniform, kaiming_normal, 
      orthogonal, 
      sparse_init,
      truncated_normal,
      identity_init,
    ]
    if init == sparse_init
      init = (args...) -> sparse_init(args...; sparsity=0.5)
    else
      # sparse_init is the only one which accepts only matrices:
      @test size(init(3)) == (3,)
      @test size(init(3, 4, 5)) == (3, 4, 5)
    end
    @test size(init(3, 4)) == (3, 4)
    # only init(size...) is accepted:
    @test_throws MethodError size(init((3, 4, 5))) == (3, 4, 5)

    # rng, and currying:
    @test size(init(MersenneTwister(1), 3, 4)) == (3, 4)
    closure = init(MersenneTwister(1))
    @test size(closure(3, 4)) == (3, 4)

    # eltype, default Float32
    @test eltype(init(3, 4)) == Float32

    # @non_differentiable
    @test gradient(x -> sum(x .* init(3, 4)), 5.0)[1] isa Number
  end

  @testset "glorot" begin
    # glorot_uniform and glorot_normal should both yield a kernel with
    # variance ≈ 2/(fan_in + fan_out)
    for dims ∈ [(1000,), (100, 100), (100, 400), (2, 3, 32, 64), (2, 3, 4, 32, 64)]
      for init ∈ [glorot_uniform, glorot_normal]
        v = init(dims...)
        fan_in, fan_out = nfan(dims...)
        σ2 = 2 / (fan_in + fan_out)
        @test 0.9σ2 < var(v) < 1.1σ2
      end
    end
  end

  @testset "kaiming" begin
    # kaiming_uniform should yield a kernel in range [-sqrt(6/n_out), sqrt(6/n_out)]
    # and kaiming_normal should yield a kernel with stddev ~= sqrt(2/n_out)
    for (n_in, n_out) in [(100, 100), (100, 400)]
      v = kaiming_uniform(n_in, n_out)
      σ2 = sqrt(6/n_out)
      @test -1σ2  < minimum(v) < -0.9σ2
      @test 0.9σ2  < maximum(v) < 1σ2

      v = kaiming_normal(n_in, n_out)
      σ2 = sqrt(2/n_out)
      @test 0.9σ2 < std(v) < 1.1σ2
    end
  end

  @testset "orthogonal" begin
    # A matrix of dim = (m,n) with m > n should produce a QR decomposition. In the other case, the transpose should be taken to compute the QR decomposition.
    for (rows,cols) in [(5,3),(3,5)]
      v = orthogonal(rows, cols)
      rows < cols ? (@test v * v' ≈ I(rows)) : (@test v' * v ≈ I(cols))
    end
    for mat in [(3,4,5),(2,2,5)]
      v = orthogonal(mat...)
      cols = mat[end]
      rows = div(prod(mat),cols)
      v = reshape(v, (rows,cols))
      rows < cols ? (@test v * v' ≈ I(rows)) : (@test v' * v ≈ I(cols))
    end
  end

  @testset "sparse_init" begin
    # sparse_init should yield an error for non 2-d dimensions
    # sparse_init should yield no zero elements if sparsity < 0
    # sparse_init should yield all zero elements if sparsity > 1
    # sparse_init should yield exactly ceil(n_in * sparsity) elements in each column for other sparsity values
    # sparse_init should yield a kernel in its non-zero elements consistent with the std parameter

    @test_throws ArgumentError sparse_init(100, 100, 100, sparsity=0.1)
    v = sparse_init(100, 100, sparsity=-0.1)
    @test sum(v .== 0) == 0
    v = sparse_init(100, 100, sparsity=1.1)
    @test sum(v .== 0) == length(v)

    for (n_in, n_out, sparsity, σ) in [(100, 100, 0.25, 0.1), (100, 400, 0.75, 0.01)]
      expected_zeros = ceil(Integer, n_in * sparsity)
      v = sparse_init(n_in, n_out, sparsity=sparsity, std=σ)
      @test all([sum(v[:,col] .== 0) == expected_zeros for col in 1:n_out])
      @test 0.9 * σ < std(v[v .!= 0]) < 1.1 * σ
    end
  end

  @testset "truncated_normal" begin
    m = truncated_normal(100, 100)
    @test minimum(m) ≈ -2 atol = 0.05  # default arguments
    @test maximum(m) ≈ 2 atol = 0.05
    @test mean(m) ≈ 0 atol = 0.1

    size100 = (100, 100, 100)
    for (μ, σ, lo, hi) in [(0.0, 1, -2, 3), (1, 2, -4.0, 5.0)]
      v = truncated_normal(size100...; mean = μ, std = σ, lo, hi)
      @test isapprox(mean(v), μ; atol = 1f-1)
      @test isapprox(minimum(v), lo; atol = 1f-2)
      @test isapprox(maximum(v), hi; atol = 1f-2)
      @test eltype(v) == Float32  # despite some Float64 arguments
    end
    for (μ, σ, lo, hi) in [(6, 2, -100.0, 100), (-7.0, 10, -100, 100)]
      v = truncated_normal(size100...; mean = μ, std = σ, lo, hi)
      @test isapprox(mean(v), μ; atol = 1f-1)
      @test isapprox(std(v), σ; atol = 1f-1)
    end
  end

  @testset "Partial application" begin
    partial_ku = kaiming_uniform(gain=1e9)
    @test maximum(partial_ku(8, 8)) > 1e9 / 2
    @test maximum(partial_ku(8, 8, gain=1)) < 1e9 / 2

    partial_kn = kaiming_normal(gain=1e9)
    @test maximum(partial_kn(8, 8)) > 1e9 / 2
    @test maximum(partial_kn(8, 8, gain=1)) < 1e9 / 2

    partial_si = sparse_init(sparsity=1)
    @test maximum(partial_si(8, 8)) == 0
    @test maximum(partial_si(8, 8, sparsity=0)) > 0
  end

  @testset "identity_init" begin

    @testset "Basic" begin
      partial = identity_init(gain=3)
      @test partial(3, 3) == identity_init(3, 3; gain=3) == [3 0 0; 0 3 0; 0 0 3]
    end

    @testset "Non-identity sizes" begin
        @test identity_init(2, 3)[:, end] == zeros(Float32, 2)
        @test identity_init(3, 2; shift=1)[1, :] == zeros(Float32, 2)
        @test identity_init(1, 1, 3, 4)[:, :, :, end] == zeros(Float32, 1, 1, 3)
        @test identity_init(2, 1, 3, 3)[end, :, :, :] == zeros(Float32, 1, 3, 3)
        @test identity_init(1, 2, 3, 3)[:, end, :, :] == zeros(Float32, 1, 3, 3)
    end

    @testset "Dense ID mapping" begin
        l = Dense(3,3, init = identity_init)

        indata = reshape(collect(Float32, 1:9), 3, 3)
        @test l(indata) == indata
    end

    @testset "$layer ID mapping with kernelsize $kernelsize" for layer in (Conv, ConvTranspose, CrossCor), kernelsize in (
        (1,),
        (3,),
        (1, 3),
        (3, 5),
        (3, 5, 7))
        nch = 3
        l = layer(kernelsize, nch=>nch, init=identity_init, pad=SamePad())

        indata = randn(Float32, kernelsize..., nch, nch)
        @test l(indata) == indata
    end

    @testset "Inception identity" begin
      insize = 7
      path1 = Conv((1, 3), insize=>2; init=identity_init, pad=SamePad())
      path2 = Conv((3, 5), insize=>3; init=identity_init(shift=(0, 0, 2, 0)), pad=SamePad())
      path3 = Conv((5, 7), insize=>2; init=identity_init(shift=(0, 0, 5, 0)), pad=SamePad())
      block = Parallel((xs...) -> cat(xs...;dims=3), path1, path2, path3)

      indata = randn(Float32, 9, 9, 7, 2)
      @test block(indata) == indata
    end
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
end

@testset "Precision" begin
  m = Chain(Dense(10, 5, relu), Dense(5, 2))
  x64 = rand(Float64, 10)
  x32 = rand(Float32, 10)
  @test eltype(m[1].weight) == Float32
  @test eltype(m(x32)) == Float32
  @test eltype(m(x64)) == Float64
  @test eltype(f64(m)(x32)) == Float64
  @test eltype(f64(m)(x64)) == Float64
  @test eltype(f64(m)[1].weight) == Float64
  @test eltype(f32(f64(m))[1].weight) == Float32
end

@testset "zero bias" begin
  m = Dense(3 => 2; bias=false)
  @test f64(m).bias === m.bias === false
  @test f32(m).bias === m.bias === false

  @testset "Gradients for broadcasted $op with sizes $s" for op in (+,-,*), s in ((1,), (2,3))
    o = ones(s)
    z = zeros(s)

    @testset "Explicit" begin
      gfun(args...) = gradient((x, y) -> sum(op.(x,y)), args...)
      g = gfun(o, z)
      @test gfun(o, false) == (g[1], nothing)

      g = gfun(z, o)
      @test gfun(false, o) == (nothing, g[2])
    end

    @testset "Implicit" begin
      gfun(args...) = gradient(() -> sum(op.(args...)), params(collect(args)))
      g = gfun(o, z)

      gres = gfun(o, false)
      @test gres[o] == g[o]
      @test false ∉ gres.params

      g = gfun(z, o)
      gres = gfun(false, o)
      @test gres[o] == g[o]
      @test false ∉ gres.params
    end
  end
end

@testset "unsqueeze" begin
  x = randn(2, 3, 2)
  @test @inferred(unsqueeze(x, dims=1)) == reshape(x, 1, 2, 3, 2)
  @test @inferred(unsqueeze(x, dims=2)) == reshape(x, 2, 1, 3, 2)
  @test @inferred(unsqueeze(x, dims=3)) == reshape(x, 2, 3, 1, 2)
  @test @inferred(unsqueeze(x, dims=4)) == reshape(x, 2, 3, 2, 1)
end

@testset "Stacking" begin
  x = randn(3,3)
  stacked = stack([x, x], dims=2)
  @test size(stacked) == (3,2,3)

  stacked_array=[ 8 9 3 5; 9 6 6 9; 9 1 7 2; 7 4 10 6 ]
  unstacked_array=[[8, 9, 9, 7], [9, 6, 1, 4], [3, 6, 7, 10], [5, 9, 2, 6]]
  @test unstack(stacked_array, dims=2) == unstacked_array
  @test stack(unstacked_array, dims=2) == stacked_array
  @test stack(unstack(stacked_array, dims=1), dims=1) == stacked_array
end

@testset "Batching" begin
  stacked_array=[ 8 9 3 5
                  9 6 6 9
                  9 1 7 2
                  7 4 10 6 ]
  unstacked_array=[[8, 9, 9, 7], [9, 6, 1, 4], [3, 6, 7, 10], [5, 9, 2, 6]]
  @test unbatch(stacked_array) == unstacked_array
  @test batch(unstacked_array) == stacked_array

  # no-op for vector of non-arrays
  @test batch([1,2,3]) == [1,2,3]
  @test unbatch([1,2,3]) == [1,2,3]

  # generic iterable
  @test batch(ones(2) for i=1:3) == ones(2, 3)
  @test unbatch(ones(2, 3)) == [ones(2) for i=1:3]
end

@testset "Param remapping" begin
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

  @testset "loadparams!" begin
    pars(w, b) = [w, b]
    pars(l) = pars(l.weight, l.bias)
    pararray(m) = mapreduce(pars, vcat, m)
    weights(m) = mapreduce(l -> [l.weight], vcat, m)
    @testset "Bias type $bt" for bt in (Flux.zeros32, nobias)
      m = dm(bt)
      loadparams!(m, params(m))
      testdense(m, bt)
    end
  end

  @testset "destructure" begin
    import Flux: destructure
    @testset "Bias type $bt" for bt in (zeros, nobias)
      m = dm(bt)
      p, re = destructure(m)
      testdense(re(p), bt)
    end

    @testset "restructure in gradient" begin
      x = rand(Float32, 3, 1)
      m = dm(zeros)
      ∇m = gradient(m -> sum(m(x)), m)[1]
      p, re = destructure(m)
      ∇p = gradient(θ -> sum(re(θ)(x)), p)[1]
      @test ∇p ≈ destructure(∇m)[1]
    end
  end
end

@testset "loadparams! & absent bias" begin
  m0 = Chain(Dense(2 => 3; bias=false, init = Flux.ones32), Dense(3 => 1))
  m1 = Chain(Dense(2 => 3; bias = Flux.randn32(3)), Dense(3 => 1))
  m2 = Chain(Dense(Float32[1 2; 3 4; 5 6], Float32[7, 8, 9]), Dense(3 => 1))

  Flux.loadparams!(m1, Flux.params(m2))
  @test m1[1].bias == 7:9
  @test sum(m1[1].weight) == 21

  # load from a model without bias -- should ideally recognise the `false` but `Params` doesn't store it
  @test_broken Flux.loadparams!(m1, Flux.params(m0))
  @test_broken iszero(m1[1].bias)
  @test sum(m1[1].weight) == 6  # written before error

  # load into a model without bias -- should it ignore the parameter which has no home, or error?
  @test_broken Flux.loadparams!(m0, Flux.params(m2))
  @test iszero(m0[1].bias)  # obviously unchanged
  @test sum(m0[1].weight) == 21
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

@testset "modules" begin
  m1 = Conv((2,3), 4=>5; pad=6, stride=7)
  m2 = LayerNorm(8)
  m3 = m2.diag
  m4 = SkipConnection(m1, +)
  m5 =  Chain(m4, m2)
  modules = Flux.modules(m5)
  # Depth-first descent
  @test length(modules) == 6
  @test modules[1] === m5
  @test modules[3] === m4
  @test modules[4] === m1
  @test modules[5] === m2
  @test modules[6] === m3

  mod_par = Flux.modules(Parallel(Flux.Bilinear(2,2,2,cbrt), Dense(2,2,abs), Dense(2,2,abs2)))
  @test length(mod_par) == 5

  mod_rnn = Flux.modules(Chain(Dense(2,3), BatchNorm(3), LSTM(3,4)))
  @test length(mod_rnn) == 6
  @test mod_rnn[end] isa Flux.LSTMCell

  mod_skip = Flux.modules(Chain(SkipConnection(
                                  Conv((2,3), 4=>5; pad=6, stride=7),
                                  +),
                                LayerNorm(8)))
  @test length(mod_skip) == 6
  @test mod_skip[end] isa Flux.Diagonal
end

@testset "Patience triggers" begin
  @testset "patience" begin
    trigger = Flux.patience(() -> true, 3)

    @test trigger() == false
    @test trigger() == false
    @test trigger() == true

    v = [false, true, false, true, true, true]
    trigger = let v = v
      Flux.patience(i -> v[i], 3)
    end

    n_iter = 0
    for i in 1:length(v)
      trigger(i) && break
      n_iter += 1
    end

    @test n_iter == 5
  end

  @testset "early stopping" begin
    @testset "args & kwargs" begin
      es = Flux.early_stopping((x; y = 1) -> x + y, 10; min_dist=3)

      n_iter = 0
      while n_iter < 99
        es(-n_iter; y=-n_iter) && break
        n_iter += 1
      end

      @test n_iter == 9
    end

    @testset "distance" begin
      es = Flux.early_stopping(identity, 10; distance=(best_score, score) -> score - best_score)

      n_iter = 0
      while n_iter < 99
        es(n_iter) && break
        n_iter += 1
      end

      @test n_iter == 99
    end

    @testset "init_score" begin
      es = Flux.early_stopping(identity, 10; init_score=10)

      n_iter = 0
      while n_iter < 99
        es(n_iter) && break
        n_iter += 1
      end

      @test n_iter == 10
    end
  end

  @testset "plateau" begin
    f = let v = 10
      () -> v = v / abs(v) - v
    end

    trigger = Flux.plateau(f, 3, init_score=10, min_dist=18)

    n_iter = 0
    while n_iter < 99
      trigger() && break
      n_iter += 1
    end

    @test n_iter == 3
  end
end

@testset "Various destructure bugs" begin

  @testset "issue 1601" begin
    struct TwoDenses
        dense::Dense
        dense2::Dense
    end
    Flux.@functor TwoDenses

    function (m::TwoDenses)(x)
        out = m.dense(x)
    end

    model = TwoDenses(
        Dense(3,1),
        Dense(3,2)
    )
    p, re = Flux.destructure(model)

    x = [1., 2., 3.]
    y, back = Flux.Zygote.pullback((x, p) -> re(p)(x), x, p)

    dy = [4.]
    dx, dp = back(dy)
    @test length(p) == length(dp)
  end

  @testset "issue 1727" begin
    p, re = Flux.destructure(BatchNorm(3))  # 6 parameters, plus 6 non-trainable
    @test length(p) == 6

    x = rand(Float32, 3, 4)
    y, back = Flux.pullback(x, p) do x, p
      vec(re(p)(x))
    end
    @test_nowarn back(y)
    b = back(y)

    @test size(b[1]) == size(x)
    @test size(b[2]) == size(p)
  end

  @testset "issue 1767" begin
    struct Model{A}
        a::A
        b::A
    end
    Flux.@functor Model
    (m::Model)(x) = m.a(x) .+ m.b(x)

    d = Dense(1, 1)
    x = rand(Float32, 1, 1)

    # Sharing the parameters
    model = Model(d, d)

    # Works
    g1 = Flux.gradient(() -> sum(model(x)), Flux.params(model))

    p, re = Flux.destructure(model)
    # Fails
    g2 = Flux.gradient(p -> sum(re(p)(x)), p)

    @test g2[1] ≈ vcat(g1[d.weight], g1[d.bias])
  end

  @testset "issue 1826" begin
    struct Split{T}  # taken from: https://fluxml.ai/Flux.jl/stable/models/advanced/#Multiple-outputs:-a-custom-Split-layer
        paths::T
    end
    Split(paths...) = Split(paths)
    Flux.@functor Split
    (m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

    n_input, n_batch, n_shared = 5, 13, 11
    n_outputs = [3, 7]

    data = rand(Float32, n_input, n_batch)
    model = Chain(
        Dense(n_input, n_shared),
        Split(Dense(n_shared, n_outputs[1]), Dense(n_shared, n_outputs[2]))
    )

    pvec, re = Flux.destructure(model)
    loss(x, idx, pv) = sum(abs2, re(pv)(x)[idx])  # loss wrt `idx`th output term

    g = Flux.Zygote.ForwardDiff.gradient(pv -> loss(data, 1, pv), pvec)
    @test g ≈ Flux.Zygote.gradient(pv -> loss(data, 1, pv), pvec)[1]
  end

end
