@testset "Pooling" begin
  x = randn(Float32, 10, 10, 3, 2)
  y = randn(Float32, 20, 20, 3, 2)
  ampx = AdaptiveMaxPool((5,5))
  @test ampx(x) == maxpool(x, PoolDims(x, 2))
  ampx = AdaptiveMeanPool((5,5))
  @test ampx(x) == meanpool(x, PoolDims(x, 2))
  ampy = AdaptiveMaxPool((10, 5))
  @test ampy(y) == maxpool(y, PoolDims(y, (2, 4)))
  ampy = AdaptiveMeanPool((10, 5))
  @test ampy(y) == meanpool(y, PoolDims(y, (2, 4)))
  gmp = GlobalMaxPool()
  @test size(gmp(x)) == (1, 1, 3, 2)
  gmp = GlobalMeanPool()
  @test size(gmp(x)) == (1, 1, 3, 2)
  mp = MaxPool((2, 2))
  @test mp(x) == maxpool(x, PoolDims(x, 2))
  mp = MeanPool((2, 2))
  @test mp(x) == meanpool(x, PoolDims(x, 2))
end

@testset "CNN" begin
  r = zeros(Float32, 28, 28, 1, 5)
  m = Chain(
    Conv((2, 2), 1 => 16, relu),
    MaxPool((2,2)),
    Conv((2, 2), 16 => 8, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10), softmax)

  @test size(m(r)) == (10, 5)

  # Test bias switch
  bias = Conv(ones(Float32, 2, 2, 1, 3), ones(Float32, 3))
  ip = zeros(Float32, 28,28,1,1)

  op = bias(ip)
  @test sum(op) == prod(size(op))

  @testset "No bias mapped through $lmap" for lmap in (identity, cpu, f32)
    model = Conv((2,2), 1=>3, bias = false) |> lmap
    op = model(ip)
    @test sum(op) ≈ 0.f0
    g = gradient(m -> sum(m(ip)), model)[1]
    @test g.bias isa Nothing
  end

  @testset "no bias train" begin
    # Train w/o bias and make sure no convergence happens
    # when only bias can be converged
    model = Conv((2, 2), 1=>3, bias = false);
    ip = zeros(Float32, 28,28,1,1)
    op = zeros(Float32, 27,27,3,1) .+ 2.f0
    opt_state = Flux.setup(Descent(), model)

    for _ = 1:10^3
      g = gradient(model) do m
        Flux.mse(m(ip), op)
      end[1]
      Flux.update!(opt_state, model, g)
    end

    @test Flux.Losses.mse(model(ip), op) ≈ 4.f0
  end

  @testset "Grouped Conv" begin
    ip = rand(Float32, 28, 100, 2)
    c = Conv((3,), 100 => 25, groups = 5)
    @test size(c.weight) == (3, 20, 25)
    @test size(c(ip)) == (26, 25, 2)

    ip = rand(Float32, 28, 28, 100, 2)
    c = Conv((3,3), 100 => 25, groups = 5)
    @test size(c.weight) == (3, 3, 20, 25)
    @test size(c(ip)) == (26, 26, 25, 2)

    ip = rand(Float32, 10, 11, 12, 100, 2)
    c = Conv((3,4,5), 100 => 25, groups = 5)
    @test size(c.weight) == (3,4,5, 20, 25)
    @test size(c(ip)) == (8,8,8, 25, 2)

    # Test that we cannot ask for non-integer multiplication factors
    @test_throws AssertionError Conv((2, 2), 3=>10, groups=2)
    @test_throws AssertionError Conv((2, 2), 2=>9, groups=2)

    # Test that Conv throws a DimensionMismatch error when the initializer 
    # produces a tensor with an incorrect shape.
    @test_throws DimensionMismatch Conv(
      (3, 3),
      1 => 1;
      init = (_...) -> rand(3, 3, 1),
    )
  end
end

@testset "_channels_in, _channels_out" begin
    _channels_in = Flux._channels_in
    _channels_out = Flux._channels_out
    @test _channels_in(Conv((3,)   , 2=>4)) == 2
    @test _channels_in(Conv((5,6,) , 2=>4)) == 2
    @test _channels_in(Conv((1,2,3), 2=>4)) == 2
    @test _channels_out(Conv((3,)   , 2=>4)) == 4
    @test _channels_out(Conv((5,6,) , 2=>4)) == 4
    @test _channels_out(Conv((1,2,3), 2=>4)) == 4

    @test _channels_in( ConvTranspose((3,)   , 1=>4)) == 1
    @test _channels_in( ConvTranspose((5,6,) , 2=>4)) == 2
    @test _channels_in( ConvTranspose((1,2,3), 3=>4)) == 3
    @test _channels_out(ConvTranspose((3,)   , 2=>1)) == 1
    @test _channels_out(ConvTranspose((5,6,) , 2=>2)) == 2
    @test _channels_out(ConvTranspose((1,2,3), 2=>3)) == 3

    @test _channels_in( ConvTranspose((6,)   , 8=>4, groups=4)) == 8
    @test _channels_in( ConvTranspose((5,6,) , 2=>4, groups=2)) == 2
    @test _channels_in( ConvTranspose((1,2,3), 3=>6, groups=3)) == 3

    @test _channels_out(ConvTranspose((1,)   , 10=>15, groups=5)) == 15
    @test _channels_out(ConvTranspose((3,2)   , 10=>15, groups=5)) == 15
    @test _channels_out(ConvTranspose((5,6,) , 2=>2, groups=2)) == 2

    for Layer in [Conv, ConvTranspose]
        for _ in 1:10
            groups = rand(1:10)
            kernel_size = Tuple(rand(1:5) for _ in rand(1:3))
            cin = rand(1:5) * groups
            cout = rand(1:5) * groups
            @test _channels_in(Layer(kernel_size, cin=>cout; groups)) == cin
            @test _channels_out(Layer(kernel_size, cin=>cout; groups)) == cout
        end
    end
end

@testset "asymmetric padding" begin
  r = ones(Float32, 28, 28, 1, 1)
  m = Conv((3, 3), 1=>1, relu; pad=(0,1,1,2))
  m.weight[:] .= 1.0
  m.bias[:] .= 0.0
  y_hat = m(r)[:,:,1,1]
  @test size(y_hat) == (27, 29)
  @test y_hat[1, 1] ≈ 6.0
  @test y_hat[2, 2] ≈ 9.0
  @test y_hat[end, 1] ≈ 4.0
  @test y_hat[1, end] ≈ 3.0
  @test y_hat[1, end-1] ≈ 6.0
  @test y_hat[end, end] ≈ 2.0
end

@testset "Depthwise Conv" begin
  r = zeros(Float32, 28, 28, 3, 5)
  m1 = DepthwiseConv((2, 2), 3=>15)
  @test size(m1(r), 3) == 15

  m2 = DepthwiseConv((2, 3), 3=>9)
  @test size(m2(r), 3) == 9

  m3 = DepthwiseConv((2, 3), 3=>9; bias=false)
  @test size(m2(r), 3) == 9

  # Test that we cannot ask for non-integer multiplication factors
  @test_throws AssertionError DepthwiseConv((2,2), 3=>10)
end

@testset "ConvTranspose" begin
  x = zeros(Float32, 5, 5, 1, 1)
  y = Conv((3,3), 1 => 1)(x)
  x_hat1 = ConvTranspose((3, 3), 1 => 1)(y)
  x_hat2 = ConvTranspose((3, 3), 1 => 1, bias=false)(y)
  @test size(x_hat1) == size(x_hat2) == size(x)

  m = ConvTranspose((3,3), 1=>1)
  # Test that the gradient call does not throw: #900
  g = gradient(m -> sum(m(x)), m)[1]

  x = zeros(Float32, 5, 5, 2, 4)
  m = ConvTranspose((3,3), 2=>3)
  g = gradient(m -> sum(m(x)), m)[1]

  # test ConvTranspose supports groups argument
  x = randn(Float32, 10, 10, 2, 3)
  m1 = ConvTranspose((3,3), 2=>4, pad=SamePad())
  @test size(m1.weight) == (3,3,4,2)
  @test size(m1(x)) == (10,10,4,3)
  m2 = ConvTranspose((3,3), 2=>4, groups=2, pad=SamePad())
  @test size(m2.weight) == (3,3,2,2)
  @test size(m1(x)) == size(m2(x))
  g = gradient(m -> sum(m(x)), m2)[1]

  x = randn(Float32, 10, 2,1)
  m = ConvTranspose((3,), 2=>4, pad=SamePad(), groups=2)
  @test size(m(x)) === (10,4,1)
  @test length(m.weight) == (3)*(2*4) / 2

  x = randn(Float32, 10, 11, 4,2)
  m = ConvTranspose((3,5), 4=>4, pad=SamePad(), groups=4)
  @test size(m(x)) === (10,11, 4,2)
  @test length(m.weight) == (3*5)*(4*4)/4

  x = randn(Float32, 10, 11, 12, 3,2)
  m = ConvTranspose((3,5,3), 3=>6, pad=SamePad(), groups=3)
  @test size(m(x)) === (10,11, 12, 6,2)
  @test length(m.weight) == (3*5*3) * (3*6) / 3

  @test occursin("groups=2", sprint(show, ConvTranspose((3,3), 2=>4, groups=2)))
  @test occursin("2 => 4"  , sprint(show, ConvTranspose((3,3), 2=>4, groups=2)))

  # test ConvTranspose outpad argument for stride > 1
  x = randn(Float32, 10, 11, 3,2)
  m1 = ConvTranspose((3,5), 3=>6, stride=3)
  m2 = ConvTranspose((3,5), 3=>6, stride=3, outpad=(1,0))
  @test size(m2(x))[1:2] == (size(m1(x))[1:2] .+ (1,0))

  x = randn(Float32, 10, 11, 12, 3,2)
  m1 = ConvTranspose((3,5,3), 3=>6, stride=3)
  m2 = ConvTranspose((3,5,3), 3=>6, stride=3, outpad=(1,0,1))
  @test size(m2(x))[1:3] == (size(m1(x))[1:3] .+ (1,0,1))
end

@testset "CrossCor" begin
  x = rand(Float32, 28, 28, 1, 1)
  w = rand(Float32, 2,2,1,1)
  y = CrossCor(w, [0.0])

  @test sum(w .* x[1:2, 1:2, :, :]) ≈ y(x)[1, 1, 1, 1]  rtol=2e-7

  r = zeros(Float32, 28, 28, 1, 5)
  m = Chain(
    CrossCor((2, 2), 1=>16, relu),
    MaxPool((2,2)),
    CrossCor((2, 2), 16=>8, relu; bias=false),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10), softmax)

  @test size(m(r)) == (10, 5)
  @test y(x) != Conv(w, [0.0])(x)
  @test CrossCor(w[end:-1:1, end:-1:1, :, :], [0.0])(x) ≈ Conv(w, [0.0])(x)  rtol=1e-7
end

@testset "Conv with non quadratic window #700" begin
  data = zeros(Float32, 7,7,1,1)
  data[4,4,1,1] = 1

  l = Conv((3,3), 1=>1)
  expected = zeros(eltype(l.weight),5,5,1,1)
  expected[2:end-1,2:end-1,1,1] = l.weight
  @test expected ≈ l(data)

  l = Conv((3,1), 1=>1)
  expected = zeros(eltype(l.weight),5,7,1,1)
  expected[2:end-1,4,1,1] = l.weight
  @test expected ≈ l(data)

  l = Conv((1,3), 1=>1)
  expected = zeros(eltype(l.weight),7,5,1,1)
  expected[4,2:end-1,1,1] = l.weight
  @test expected ≈ l(data)

  @test begin
    # we test that the next expression does not throw
    randn(Float32, 10,10,1,1) |> Conv((6,1), 1=>1, Flux.σ)
    true
  end
end

@testset "$ltype $(nd)D symmetric non-constant padding" for ltype in (Conv, ConvTranspose, DepthwiseConv, CrossCor), nd in (1, 2, 3)
  kernel = ntuple(Returns(3), nd)
  data = ones(Float32, (kernel .+ 5)..., 1,1)

  pad = ntuple(i -> i, nd)
  l = ltype(kernel, 1=>1, pad=pad)

  expanded_pad = ntuple(i -> pad[(i - 1) ÷ 2 + 1], 2 * nd)
  l_expanded = ltype(kernel, 1=>1, pad=expanded_pad)

  @test size(l(data)) == size(l_expanded(data))
end

@testset "$ltype SamePad kernelsize $k" for ltype in (Conv, ConvTranspose, DepthwiseConv, CrossCor), k in ( (1,), (2,), (3,), (4,5), (6,7,8))
  data = ones(Float32, (k .+ 3)..., 1,1)
  l = ltype(k, 1=>1, pad=SamePad())
  @test size(l(data)) == size(data)

  l = ltype(k, 1=>1, pad=SamePad(), dilation = k .÷ 2)
  @test size(l(data)) == size(data)

  stride = 3
  l = ltype(k, 1=>1, pad=SamePad(), stride = stride)
  if ltype == ConvTranspose
    @test size(l(data))[1:end-2] == stride .* size(data)[1:end-2]
  else
    @test size(l(data))[1:end-2] == cld.(size(data)[1:end-2], stride)
  end
end

@testset "$ltype SamePad windowsize $k" for ltype in (MeanPool, MaxPool), k in ( (1,), (2,), (3,), (4,5), (6,7,8))
  data = ones(Float32, (k .+ 3)..., 1,1)

  l = ltype(k, pad=SamePad())
  @test size(l(data))[1:end-2] == cld.(size(data)[1:end-2], k)
end

@testset "bugs fixed" begin
  # https://github.com/FluxML/Flux.jl/issues/1421
  @test Conv((5, 5), 10 => 20, identity; init = Base.randn).bias isa Vector{Float64}
end

@testset "constructors: $fun" for fun in [Conv, CrossCor, ConvTranspose, DepthwiseConv]
  @test fun(rand(2,3,4)).bias isa Vector{Float64}
  @test fun(rand(2,3,4,5), false).bias === false
  if fun == Conv
    @test fun(rand(2,3,4,5,6), rand(6)).bias isa Vector{Float64}
    @test_skip fun(rand(2,3,4,5,6), 1:6).bias isa Vector{Float64}
  elseif fun == DepthwiseConv
    @test fun(rand(2,3,4,5,6), rand(30)).bias isa Vector{Float64}
  end
  @test_throws DimensionMismatch fun(rand(2,3,4), rand(6))
end

@testset "type matching" begin
  x = rand(Float64, 10,2,5)
  xi = rand(-3:3, 10,2,5)
  c1 = Conv((3,), 2=>4, relu)
  @test @inferred(c1(x)) isa Array{Float32, 3}
  @test c1(xi) isa Array{Float32, 3}

  c2 = CrossCor((3,), 2=>1, relu)
  @test @inferred(c2(x)) isa Array{Float32, 3}

  c3 = ConvTranspose((3,), 2=>4, relu)
  @test c3(x) isa Array{Float32, 3}
  @test (@inferred c3(x); true)
end
