using Flux
using Flux: throttle, nfan, glorot_uniform, glorot_normal, kaiming_normal, kaiming_uniform, stack, unstack, Zeros
using StatsBase: var, std
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

  @testset "glorot" begin
    # glorot_uniform and glorot_normal should both yield a kernel with
    # variance ≈ 2/(fan_in + fan_out)
    for dims ∈ [(1000,), (100, 100), (100, 400), (2, 3, 32, 64), (2, 3, 4, 32, 64)]
      for init ∈ [glorot_uniform, glorot_normal]
        v = init(dims...)
        fan_in, fan_out = nfan(dims...)
        σ2 = 2 / (fan_in + fan_out)
        @test 0.9σ2 < var(v) < 1.1σ2
        @test eltype(v) == Float32
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
      @test eltype(v) == Float32

      v = kaiming_normal(n_in, n_out)
      σ2 = sqrt(2/n_out)
      @test 0.9σ2 < std(v) < 1.1σ2
      @test eltype(v) == Float32
    end
  end
end

@testset "Params" begin
  m = Dense(10, 5)
  @test size.(params(m)) == [(5, 10), (5,)]
  m = RNN(10, 5)
  @test size.(params(m)) == [(5, 10), (5, 5), (5,), (5,)]

  # Layer duplicated in same chain, params just once pls.
  c = Chain(m, m)
  @test size.(params(c)) == [(5, 10), (5, 5), (5,), (5,)]

  # Self-referential array. Just want params, no stack overflow pls.
  r = Any[nothing,m]
  r[1] = r
  @test size.(params(r)) == [(5, 10), (5, 5), (5,), (5,)]
end

@testset "Basic Stacking" begin
  x = randn(3,3)
  stacked = stack([x, x], 2)
  @test size(stacked) == (3,2,3)
end

@testset "Precision" begin
  m = Chain(Dense(10, 5, relu), Dense(5, 2))
  x = rand(10)
  @test eltype(m[1].W) == Float32
  @test eltype(m(x)) == Float32
  @test eltype(f64(m)(x)) == Float64
  @test eltype(f64(m)[1].W) == Float64
  @test eltype(f32(f64(m))[1].W) == Float32
end

@testset "Zeros" begin
  m = Dense(randn(2,3), Zeros())
  @test f64(m).b === m.b === Zeros()
  @test f32(m).b === m.b === Zeros()

  @testset "Gradients for broadcasted $op with sizes $s" for op in (+,-,*), s in ((1,), (2,3))
    o = ones(s)
    z = zeros(s)
    Zs = Zeros(s...)
    Z0 = Zeros()

    @testset "Explicit" begin
      gfun(args...) = gradient((x, y) -> sum(op.(x,y)), args...)
      g = gfun(o, z) 
      @test gfun(o, Zs) == (g[1], nothing)
      @test gfun(o, Z0) == (g[1], nothing)

      g = gfun(z, o) 
      @test gfun(Zs, o) == (nothing, g[2])
      @test gfun(Z0, o) == (nothing, g[2])

      @test gfun(Zs, Zs) == gfun(Z0, Z0) == gfun(Zs, Z0) == gfun(Z0, Zs) == (nothing, nothing)
    end
    
    @testset "Implicit" begin
      gfun(args...) = gradient(() -> sum(op.(args...)), params(collect(args)))
      g = gfun(o, z) 

      gres = gfun(o, Zs)
      @test gres[o] == g[o]
      @test gres[Zs] === nothing

      gres = gfun(o, Z0)
      @test gres[o] == g[o]
      @test gres[Z0] === nothing

      g = gfun(z, o) 

      gres = gfun(Zs, o)
      @test gres[o] == g[o]
      @test gres[Zs] === nothing


      gres = gfun(Z0, o)
      @test gres[o] == g[o]
      @test gres[Z0] === nothing

      gfunc(args...) = gfun(args...).grads |> values |> Tuple
      @test gfunc(Zs, Zs) == gfunc(Z0, Z0) == (nothing,)
      @test gfunc(Zs, Z0) == gfunc(Z0, Zs) == (nothing, nothing)
    end
  end

  @testset "Gradients for broadcasted / with sizes $s" for s in ((1,), (2,3))
    o = ones(s)
    z = zeros(s)
    Z = Zeros() # Only defined for 0-dim

    @testset "Explicit" begin
      gfun(args...) = gradient((x, y) -> sum(x ./ y), args...)
      g = gfun(z, o) 
      @test gfun(Z, o) == (nothing, g[2])
    end

    @testset "Implicit" begin
      gfun(x,y) = gradient(() -> sum(x ./ y), params([x,y]))
      
      g = gfun(z, o) 
      gres = gfun(Z, o)
      @test gres[o] == g[o]
      @test gres[Z] === nothing
    end
  end

  @testset "Gradients for $op with sizes $s" for op in (+,-), s in (tuple(), (1,), (2,3))
    o = ones(s)
    z = zeros(s)
    Z = Zeros(s...)


    @testset "Explicit" begin
      gfun(args...) = gradient((x, y) -> sum(op(x,y)), args...)

      g = gfun(o, z) 
      @test gfun(o, Z) == (g[1], nothing)

      g = gfun(z, o) 
      @test gfun(Z, o) == (nothing, g[2])

      @test gfun(Z, Z) == (nothing, nothing)
    end

    @testset "Implicit" begin
      gfun(args...) = gradient(() -> sum(op(args...)), params(collect(args)))
      g = gfun(o, z) 
      gres = gfun(o, Z)
      @test gres[o] == g[o]
      @test gres[Z] === nothing

      g = gfun(z, o) 
      gres = gfun(Z, o)
      @test gres[o] == g[o]
      @test gres[Z] === nothing

      gfunc(args...) = gfun(args...).grads |> values |> Tuple
      @test gfunc(Z, Z) == (nothing,)
    end
  end
end

@testset "Stacking" begin
  stacked_array=[ 8 9 3 5; 9 6 6 9; 9 1 7 2; 7 4 10 6 ]
  unstacked_array=[[8, 9, 9, 7], [9, 6, 1, 4], [3, 6, 7, 10], [5, 9, 2, 6]]
  @test unstack(stacked_array, 2) == unstacked_array
  @test stack(unstacked_array, 2) == stacked_array
  @test stack(unstack(stacked_array, 1), 1) == stacked_array
end

@testset "Param remapping" begin
  ls(dims...) = reshape(collect(Float32, 1:prod(dims)), dims...)
  dl(nin, nout, bias) = Dense(ls(nin, nout), bias(nout)) 
  dm(bias) = Chain(
    dl(3, 5, bias),
    dl(5, 4, bias),
    dl(4, 3, bias)
  )

  testdense(m, bt) = @testset "Check layer $i" for (i, (l1, l2)) in enumerate(zip(m, dm(bt)))
    @test l1.W == l2.W
    @test l1.b == l2.b
    @test typeof(l1.b) === typeof(l2.b)
  end

  ZerosNoShape(::Integer) = Zeros() # Just to get a readable name in testsets
  @testset "loadparams!" begin 
    import Flux: loadparams!
    pararray(m) = mapreduce(l -> collect(params(l).order), vcat, m)
    @testset "Bias type $bt" for bt in (zeros, Zeros, ZerosNoShape)
      m = dm(bt)
      loadparams!(m, params(m))
      testdense(m, bt)
    end

    @testset "$b1 to $b2" for (b1, b2, be) in (
      (zeros, ones, ones),          # Load ones as bias to a model with zeros as bias -> model gets ones as bias
      (ones, Zeros, zeros),         # Load Zeros as bias to a model with ones as bias-> model gets zeros as bias
      (Zeros, ones, Zeros),         # Load ones as bias to a model with Zeros as bias-> model bias does not change
      (ones, ZerosNoShape, zeros),  # Load 0d Zeros as bias to a model with ones as bias -> model get zeros as bias   
      #(ZerosNoShape, ones, ZerosNoShape), # Does not work as loadmodel! uses params which is backed by a set -> different number of parameters in models
    )
      m1 = dm(b1)
      m2 = dm(b2)
      loadparams!(m1, pararray(m2))
      testdense(m1, be)
    end
  end

  @testset "destructure" begin
    import Flux: destructure
    @testset "Bias type $bt" for bt in (zeros, Zeros, ZerosNoShape)
      m = dm(bt)
      p, re = destructure(m)
      testdense(re(p), bt)
    end
  end
end