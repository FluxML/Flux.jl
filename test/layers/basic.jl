using Test, Random
using Flux: activations

@testset "basic" begin
  @testset "helpers" begin
    @testset "activations" begin
      dummy_model = Chain(x->x.^2, x->x .- 3, x -> tan.(x))
      x = randn(10)
      @test activations(dummy_model, x)[1] == x.^2
      @test activations(dummy_model, x)[2] == (x.^2 .- 3)
      @test activations(dummy_model, x)[3] == tan.(x.^2 .- 3)

      @test activations(Chain(), x) == ()
      @test activations(Chain(identity, x->:foo), x)[2] == :foo # results include `Any` type
    end
  end

  @testset "Chain" begin
    @test_nowarn Chain(Dense(10, 5, σ), Dense(5, 2))(randn32(10))
    @test_throws DimensionMismatch Chain(Dense(10, 5, σ),Dense(2, 1))(randn32(10))
    # numeric test should be put into testset of corresponding layer

    @test_nowarn Chain(first = Dense(10, 5, σ), second = Dense(5, 2))(randn32(10))
    m = Chain(first = Dense(10, 5, σ), second = Dense(5, 2))
    @test m[:first] == m[1]
    @test m[1:2] == m

    @test m == m
    @test m == fmap(identity, m)  # does not forget names

    @test_throws ArgumentError Chain(layers = Dense(10, 10), two = identity) # reserved name

    @test_nowarn Chain([Dense(10, 5, σ), Dense(5, 2)])(randn(Float32, 10))  # vector of layers
    
    c = Chain(Dense(10, 5, σ), Dense(5, 2), Dense(2, 1, relu))
    @test c[1] == c[begin]
    @test c[3] == c[end]

    @test Chain(identity)(1,2,3) == (1,2,3)  # multiple args become a tuple
  end

  @testset "Activations" begin
    c = Chain(Dense(3,5,relu), Dense(5,1,relu))
    X = Float32.([1.0; 1.0; 1.0])
    @test_nowarn gradient(()->Flux.activations(c, X)[2][1], Flux.params(c))

    c2 = Chain(enc = c[1], dec = c[2])
    @test Flux.activations(c, X) == Flux.activations(c2, X)
    @test_nowarn gradient(()->Flux.activations(c2, X)[2][1], Flux.params(c2))
  end

  @testset "Dense" begin
    @testset "constructors" begin
      @test size(Dense(10, 100).weight) == (100, 10)
      @test size(Dense(10, 100).bias) == (100,)
      @test Dense(rand(100,10), rand(100)).σ == identity
      @test Dense(rand(100,10)).σ == identity

      @test Dense(rand(100,10), false).σ == identity
      @test Dense(rand(100,10), false, tanh).σ == tanh
      @test Dense(rand(100,10), rand(100)).σ == identity
      @test Dense(rand(Float16, 100,10), true).bias isa Vector{Float16}  # creates matching type
      @test Dense(rand(Float16, 100,10), rand(100)).bias isa Vector{Float16}  # converts to match

      @test Dense(3,4; init=Base.randn, bias=true).bias isa Vector{Float64}
      @test Dense(3,4; init=Base.randn, bias=[1,2,3,4]).bias isa Vector{Float64}

      @test_throws MethodError Dense(10, 10.5)
      @test_throws MethodError Dense(10, 10.5, tanh)
      @test_throws DimensionMismatch Dense(3,4; bias=rand(5))
      @test_throws DimensionMismatch Dense(rand(4,3), rand(5))
      @test_throws MethodError Dense(rand(5))
      @test_throws MethodError Dense(rand(5), rand(5))
      @test_throws MethodError Dense(rand(5), rand(5), tanh)
    end
    @testset "dimensions" begin
      @test  length(Dense(10 => 5)(randn32(10))) == 5
      @test_throws DimensionMismatch Dense(10 => 5)(randn32(1))
      @test_throws MethodError Dense(10 => 5)(1) # avoid broadcasting
      @test_throws MethodError Dense(10 => 5).(randn32(10)) # avoid broadcasting
      @test size(Dense(10, 5)(randn(10))) == (5,)
      @test size(Dense(10, 5)(randn(10,2))) == (5,2)
      @test size(Dense(10, 5)(randn(10,2,3))) == (5,2,3)
      @test size(Dense(10, 5)(randn(10,2,3,4))) == (5,2,3,4)
      @test_throws DimensionMismatch Dense(10, 5)(randn(11,2,3))
    end
    @testset "zeros" begin
      @test Dense(10, 1, identity, init = ones)(ones(10,1)) == 10*ones(1, 1)
      @test Dense(10, 1, identity, init = ones)(ones(10,2)) == 10*ones(1, 2)
      @test Dense(10, 2, identity, init = ones)(ones(10,1)) == 10*ones(2, 1)
      @test Dense(10, 2, identity, init = ones)([ones(10,1) 2*ones(10,1)]) == [10 20; 10 20]
      @test Dense(10, 2, identity, init = ones, bias = false)([ones(10,1) 2*ones(10,1)]) == [10 20; 10 20]
    end
    @testset "type matching" begin
       d1 = Dense(2 => 3)
       d2 = Dense(d1.weight, false)
       x1 = randn(Float32, 2, 4)
       @test d1(x1) ≈ d2(x1) ≈ d1.weight * x1
       x2 = Float64.(x1)
       @test d1(x2) ≈ d2(x2) ≈ d1.weight * x2
       @test d1(x2) isa Array{Float32}  # tests _match_eltype, will print a warning
       @test d2(x2) isa Array{Float32}

       x3 = rand(-5:5, 2, 4)
       @test d1(x3) ≈ d2(x3) ≈ d1.weight * x3
       x4 = rand(Bool, 2, 4)
       @test d1(x4) ≈ d2(x4) ≈ d1.weight * x4
       x5 = Flux.onehotbatch(rand(Bool, 5), (true, false))
       @test d1(x5) ≈ d2(x5) ≈ d1.weight * x5
     end
  end

  @testset "Scale" begin
    @test length(Flux.Scale(10)(randn(10))) == 10
    @test length(Flux.Scale(10)(randn(1))) == 10
    @test length(Flux.Scale(10; bias = false)(randn(10))) == 10
    @test length(Flux.Scale(10, tanh)(randn(10))) == 10
    @test_throws DimensionMismatch Flux.Scale(10)(randn(2))

    @test Flux.Scale(2)([1 2]) == [1 2; 1 2]
    @test Flux.Scale(2)([1, 2]) == [1, 2]
    @test Flux.Scale(2; init = randn)([1, 2]) != [1, 2]
    @test Flux.Scale(2; bias = false)([1 2; 3 4]) == [1 2; 3 4]
    @test Flux.Scale(2, abs2; bias = false, init = ones)([1 2; 3 4]) == [1 4; 9 16]

    @test Flux.Scale(2)(rand(2, 3, 4)) |> size == (2, 3, 4)
    @test Flux.Scale(2, 3;)(rand(2, 3, 4)) |> size == (2, 3, 4)
    @test Flux.Scale(2, 3, 4; bias = false)(rand(2, 3, 4)) |> size == (2, 3, 4)
    @test Flux.Scale(2, 3; bias = false)(rand(2, 1, 4)) |> size == (2, 3, 4)
    @test Flux.Scale(2, 3, tanh; bias = false, init = zeros)(rand(2, 1, 4)) == zeros(2, 3, 4)
    
    @test_throws MethodError Flux.Scale(1.)
    @test_throws MethodError Flux.Scale(1., 2.)
    @test_throws Exception Flux.Scale()
    @test_throws MethodError Flux.Scale(sin)
  end

  @testset "Maxout" begin
    # Note that the normal common usage of Maxout is as per the docstring
    # These are abnormal constructors used for testing purposes

    @testset "Constructor" begin
      mo = Maxout(() -> identity, 4)
      input = rand(40)
      @test mo(input) == input
    end

    @testset "simple alternatives" begin
      mo = Maxout(x -> x, x -> 2x, x -> 0.5x)
      input = rand(40)
      @test mo(input) == 2*input
    end

    @testset "complex alternatives" begin
      mo = Maxout(x -> [0.5; 0.1]*x, x -> [0.2; 0.7]*x)
      input = [3.0 2.0]
      target = [0.5, 0.7].*input
      @test mo(input) == target
    end

    @testset "params" begin
      mo = Maxout(()->Dense(32, 64), 4)
      ps = Flux.params(mo)
      @test length(ps) == 8  #4 alts, each with weight and bias
    end
  end

  @testset "SkipConnection" begin
    @testset "zero sum" begin
      input = randn(10, 10, 10, 10)
      @test SkipConnection(x -> zeros(size(x)), (a,b) -> a + b)(input) == input
    end

    @testset "concat size" begin
      input = randn(10, 2)
      @test size(SkipConnection(Dense(10,10), (a,b) -> cat(a, b, dims = 2))(input)) == (10,4)
    end
  end

  @testset "Bilinear" begin
    @testset "SkipConnection recombinator" begin
      d = Dense(10, 10)
      b = Flux.Bilinear(10, 10, 5)
      x = randn(Float32,10,9)
      sc = SkipConnection(d, b)
      @test size(sc(x)) == (5,9)
    end

    @testset "Two-streams zero sum" begin
      x = zeros(Float32,10,9)
      y = zeros(Float32,2,9)
      b = Flux.Bilinear(10, 2, 3)
      @test size(b(x,y)) == (3,9)
      @test sum(abs2, b(x,y)) == 0f0
    end

    @testset "Inner interactions" begin
      x = randn(Float32,11,7)
      b = Flux.Bilinear(11, 11, 3)
      @test size(b(x)) == (3,7)
      @test_nowarn gs = gradient(() -> sum(abs2.(b(x))), params(b))
    end

    @testset "constructors" begin
      b1 = Flux.Bilinear(randn(3,4,5))
      @test b1.bias isa Vector{Float64}
      @test b1.σ == identity

      b2 = Flux.Bilinear(randn(3,4,5), false)
      @test b2.bias === false

      b3 = Flux.Bilinear(randn(Float16, 3,4,5), true, tanh)
      @test b3.σ == tanh
      @test b3.bias isa Vector{Float16}
      @test size(b3(rand(4), rand(5))) == (3,)

      b4 = Flux.Bilinear(3,3,7; bias=1:7, init=Flux.zeros32)
      @test_skip  b4.bias isa Vector{Float32}

      @test_throws ArgumentError Flux.Bilinear(rand(3)) # expects a 3-array
      @test_throws ArgumentError Flux.Bilinear(rand(3,4), false, tanh)
      @test_throws DimensionMismatch Flux.Bilinear(rand(3,4,5), rand(6), tanh) # wrong length bias
    end
  end

  @testset "Parallel" begin
    @testset "zero sum" begin
      input = randn(10, 10, 10, 10)
      @test Parallel(+, x -> zeros(size(x)), identity)(input) == input
    end

    @testset "concat size" begin
      input = randn(10, 2)
      @test size(Parallel((a, b) -> cat(a, b; dims=2), Dense(10, 10), identity)(input)) == (10, 4)
      @test size(Parallel(hcat, one = Dense(10, 10), two = identity)(input)) == (10, 4)
    end

    @testset "vararg input" begin
      inputs = randn32(10), randn32(5), randn32(4)
      @test size(Parallel(+, Dense(10, 2), Dense(5, 2), Dense(4, 2))(inputs)) == (2,)
      @test size(Parallel(+; a = Dense(10, 2), b = Dense(5, 2), c = Dense(4, 2))(inputs)) == (2,)
      @test_throws ArgumentError Parallel(+, sin, cos)(1,2,3)  # wrong number of inputs
      @test Parallel(+, sin, cos)(pi/2) ≈ 1  # one input, several layers
      @test Parallel(/, abs)(3, -4) ≈ 3/4    # one layer, several inputs
      @test Parallel(/, abs)((3, -4)) ≈ 3/4
      @test Parallel(/; f=abs)(3, -4) ≈ 3/4
    end

    @testset "named access" begin
      m = Parallel(hcat, one = Dense(10, 10), two = identity)
      @test m[1] == m[:one]
      @test m[1:2] == m

      @test_throws ArgumentError Parallel(hcat, layers = Dense(10, 10), two = identity) # reserved names
      @test_throws ArgumentError Parallel(hcat, connection = Dense(10, 10), two = identity)

      @test m == fmap(identity, m)  # does not forget names

      @test Parallel(vcat, x = log)(1) == [0]
      @test Parallel(vcat, log)(1) == [0]
    end

    @testset "trivial cases" begin
      @test Parallel(hcat) isa Parallel{typeof(hcat), Tuple{}}  # not a NamedTuple
      @test Parallel(hcat)(1) == hcat()
      @test Parallel(hcat, inv)(2) == hcat(1/2)  # still calls connection once.
    end

    @testset "connection is called once" begin
      CNT = Ref(0)
      f_cnt = (x...) -> (CNT[]+=1; +(x...))
      Parallel(f_cnt, sin, cos, tan)(1)
      @test CNT[] == 1
      Parallel(f_cnt, sin, cos, tan)(1,2,3)
      @test CNT[] == 2
      Parallel(f_cnt, sin)(1)
      @test CNT[] == 3
      Parallel(f_cnt, sin)(1,2,3)
      @test CNT[] == 4
    end

    # Ref https://github.com/FluxML/Flux.jl/issues/1673
    @testset "Input domain" begin
      struct Input
        x
      end

      struct L1
        x
      end
      (l::L1)(x) = l.x * x
      Flux.@functor L1
      Base.:*(a::AbstractArray, b::Input) = a * b.x

      par = Parallel(+, L1(rand(Float32, 3,3)), L1(rand(Float32, 3,3)))
      ip = Input(rand(Float32, 3,3))
      ip2 = Input(rand(Float32, 3,3))

      @test par(ip) ≈ par.layers[1](ip.x) + par.layers[2](ip.x)
      @test par(ip, ip2) ≈ par.layers[1](ip.x) + par.layers[2](ip2.x)
      gs = gradient((par, x...) -> sum(par(x...)), par, ip, ip2)
      gs_reg = gradient(par, ip, ip2) do par, x, y
        sum(par.layers[1](x.x) + par.layers[2](y.x))
      end

      for (a,b) in zip(gs[1].layers, gs_reg[1].layers)
        @test a.x ≈ b.x
      end
      @test gs[2].x ≈ gs_reg[2].x
      @test gs[3].x ≈ gs_reg[3].x
    end
  end

  @testset "Embedding" begin
    vocab_size, embed_size = 10, 4
    m = Embedding(vocab_size, embed_size)
    @test size(m.weight) == (embed_size, vocab_size)
    
    # one index
    @test m(1) isa Vector{Float32}
    @test m(2) ≈ m.weight[:,2]
    @test m(OneHotVector(3, vocab_size)) ≈ m.weight[:,3]
    @test_throws DimensionMismatch m(OneHotVector(3, 1000))
    @test m(4) ≈ m((1:vocab_size) .== 4)

    # a batch of indices
    x = rand(1:vocab_size, 3)
    y = m(x)
    @test y isa Matrix{Float32}
    @test y ≈ m.weight[:,x]
    x2 = OneHotMatrix(x, vocab_size)
    y2 = m(x2)
    @test y2 isa Matrix{Float32}
    @test y2 ≈ y
    @test_throws DimensionMismatch m(OneHotMatrix(x, 1000))
    @test y ≈ m(x' .== (1:vocab_size))

    # more dimensions via reshape
    x = rand(1:vocab_size, 3, 4)
    y = m(x)
    @test y isa Array{Float32, 3}
    @test size(y) == (embed_size, 3, 4)
    x3 = Flux.onehotbatch(x, 1:1:vocab_size)
    @test size(x3) == (vocab_size, 3, 4)
    y3 = m(x3)
    @test size(y3) == (embed_size, 3, 4)
  end

  @testset "EmbeddingBag" begin

    # test _splitat
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    offsets_good = [1, 3, 6]
    offsets_each = [1,2,3,4,5,6,7,8,9]
    offsets_just_one = [1]
    offsets_all_but_last = [1, 9]

    @test Flux._splitat(data, offsets_good) == [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
    @test Flux._splitat(data, offsets_each) == [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
    @test Flux._splitat(data, offsets_just_one) == [[1,2,3,4,5,6,7,8,9]]
    @test Flux._splitat(data, offsets_all_but_last) == [[1,2,3,4,5,6,7,8], [9]]

    offsets_non_monotonic = [1, 2, 2, 5]
    offsets_non_sorted = [1, 5, 2]
    offsets_non_one = [2, 3, 5]
    offsets_too_large = [1, 5, 11]

    @test_throws ArgumentError  Flux._splitat(data, offsets_non_monotonic)
    @test_throws ArgumentError  Flux._splitat(data, offsets_non_sorted)
    @test_throws ArgumentError  Flux._splitat(data, offsets_non_one)
    @test_throws ArgumentError  Flux._splitat(data, offsets_too_large)

    @testset for reduction in [sum, Statistics.mean, maximum]
      vocab_size, embed_size = 10, 4
      emb_bag = Flux.EmbeddingBag(vocab_size => embed_size, reduction)
      emb = Flux.Embedding(emb_bag.weight)
      @test size(emb_bag.weight) == (embed_size, vocab_size)
      @test_throws ErrorException emb_bag(2)

      # single bag (input as a vector)
      x = rand(1:vocab_size, 3)
      y = emb_bag(x)
      z = vec(reduction(emb(x), dims=2))
      @test y isa Vector{Float32}
      @test y ≈ z

      # PyTorch style `input`/`offset` bagging
      @test emb_bag([1,3,2,4,5,7], [1,3,5]) ≈ emb_bag([[1,3], [2,4], [5,7]])
      @test emb_bag([1,3,2,4,5,7], [1,3,5]) ≈ emb_bag([1 2 5; 3 4 7])
      @test_throws ArgumentError emb_bag([1,2,3,4,5,6], [2, 4])
      @test_throws ArgumentError emb_bag([1,2,3,4,5,6], [1, 12])

      # docstring example
      @test emb_bag([1,2,3,4,5,6,7,8,9,10], [1,5,6,8]) ≈ emb_bag([[1,2,3,4], [5], [6,7], [8,9,10]])

      # multiple bags (input as a vector of vectors)
      x = [rand(1:vocab_size, 3) for _ in 1:4]
      y = emb_bag(x)
      z = reduce(hcat, reduction.(emb.(x), dims=2))
      @test y isa Matrix{Float32}
      @test y ≈ z

      # multiple bags (input as a matrix)
      x = rand(1:vocab_size, (3, 5))
      xvec = collect(eachcol(x))
      y = emb_bag(x)
      z = reduce(hcat, reduction.(emb.(xvec), dims=2))
      @test y ≈ emb_bag(xvec)
      @test y ≈ z

      # a one-hot matrix is a bag, but a one-hot vector is not.
      @test_throws ErrorException emb_bag(Flux.OneHotVector(3, vocab_size))

      i2 = rand(1:vocab_size, 3)
      x2 = Flux.OneHotMatrix(i2, vocab_size)
      y2 = emb_bag(x2)
      z2 = emb(i2)
      @test y2 isa Vector{Float32}
      @test y2 ≈ vec(reduction(z2, dims=2))
      @test_throws DimensionMismatch emb_bag(Flux.OneHotMatrix(1:5, 1000))
    end
  end
end

@testset "second derivatives" begin
  m1 = Chain(Dense(3,4,tanh; bias=false), Dense(4,2))
  @test Zygote.hessian_dual(sum∘m1, [1,2,3]) ≈ Zygote.hessian_reverse(sum∘m1, [1,2,3])

  m1v = Chain([m1[1], m1[2]])  # vector of layers
  @test Zygote.hessian_dual(sum∘m1v, [1,2,3]) ≈ Zygote.hessian_dual(sum∘m1, [1,2,3])
  @test Zygote.hessian_dual(sum∘m1v, [1,2,3]) ≈ Zygote.hessian_reverse(sum∘m1v, [1,2,3])

  # NNlib's softmax gradient writes in-place
  m2 = Chain(Dense(3,4,tanh), Dense(4,2), softmax)
  @test_broken Zygote.hessian_dual(sum∘m2, [1,2,3]) ≈ Zygote.hessian_reverse(sum∘m2, [1,2,3])

  # https://github.com/FluxML/NNlib.jl/issues/362
  m3 = Chain(Conv((3,), 2 => 3, relu), Dense(2,2))
  x3 = cat(Float32[1 2; 3 4; 5 6; 7 8]; dims=3)
  @test Zygote.hessian_dual(sum∘m3, x3) ≈ Zygote.hessian_reverse(sum∘m3, x3)
end

@testset "gradients of Chain{Vector}" begin
  m1 = Chain(Dense(3,4,tanh; bias=false), Dense(4,2))
  m1v = Chain([m1[1], m1[2]])
  @test sum(length, params(m1)) == sum(length, params(m1v))

  x1 = randn(Float32,3,5)
  @test m1(x1) ≈ m1v(x1)

  y1 = rand(Bool,2,5)
  g1 = gradient(() -> Flux.Losses.logitcrossentropy(m1(x1), y1), params(m1))
  g1v = gradient(() -> Flux.Losses.logitcrossentropy(m1v(x1), y1), params(m1v))
  @test g1[m1[1].weight] ≈ g1v[m1v[1].weight]
  @test g1[m1[2].bias] ≈ g1v[m1v[2].bias]

  @test Flux.destructure(m1)[1] ≈ Flux.destructure(m1v)[1]
  z1 = rand(22);
  @test Flux.destructure(m1)[2](z1)[1].weight ≈ Flux.destructure(m1v)[2](z1)[1].weight
  # Note that Flux.destructure(m1v)[2](z) has a Chain{Tuple}, as does m1v[1:2]
end

@testset "PairwiseFusion" begin
  x = (rand(1, 10), rand(30, 10))
  layer = PairwiseFusion(+,  Dense(1, 30),  Dense(30, 10))
  y = layer(x)
  @test length(y) == 2
  @test size(y[1]) == (30, 10)
  @test size(y[2]) == (10, 10)

  x = rand(1, 10)
  layer = PairwiseFusion(.+,  Dense(1, 10),  Dense(10, 1))
  y = layer(x)
  @test length(y) == 2
  @test size(y[1]) == (10, 10)
  @test size(y[2]) == (1, 10)

  @test PairwiseFusion(vcat, x->x.+1, x->x.+2, x->x.^3)(2, 10, 20) == (3, [5, 12], [125, 1728, 8000])
  @test PairwiseFusion(vcat, x->x.+1, x->x.+2, x->x.^3)(7) == (8, [10, 9], [1000, 729, 343])
end
