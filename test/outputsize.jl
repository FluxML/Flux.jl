@testset "basic" begin
  m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32))
  @test outputsize(m, (10, 10, 3, 1)) == (6, 6, 32, 1)

  m = Dense(10, 5)
  @test_throws DimensionMismatch outputsize(m, (5, 2)) == (5, 1)
  @test outputsize(m, (10,); padbatch=true) == (5, 1)

  m = Chain(Dense(10, 8, σ), Dense(8, 5), Dense(5, 2))
  @test outputsize(m, (10,); padbatch=true) == (2, 1)
  @test outputsize(m, (10, 30)) == (2, 30)

  @info "Don't mind the following error, it's for testing purpose."
  m = Chain(Dense(10, 8, σ), Dense(8, 4), Dense(5, 2))
  @test_throws DimensionMismatch outputsize(m, (10,))

  m = Flux.Diagonal(10)
  @test outputsize(m, (10, 1)) == (10, 1)

  m = Maxout(() -> Conv((3, 3), 3 => 16), 2)
  @test outputsize(m, (10, 10, 3, 1)) == (8, 8, 16, 1)

  m = flatten
  @test outputsize(m, (5, 5, 3, 10)) == (75, 10)

  m = Flux.unsqueeze(dims=3)
  @test outputsize(m, (5, 7, 13)) == (5, 7, 1, 13)

  m = Flux.Bilinear(10, 10, 7)
  @test outputsize(m, (10,)) == (7,)
  @test outputsize(m, (10, 32)) == (7, 32)

  m = Chain(Conv((3, 3), 3 => 16), BatchNorm(16), flatten, Dense(1024, 10))
  @test outputsize(m, (10, 10, 3, 50)) == (10, 50)
  @test outputsize(m, (10, 10, 3, 2)) == (10, 2)

  m = SkipConnection(Conv((3, 3), 3 => 16; pad = 1), (mx, x) -> cat(mx, x; dims = 3))
  @test outputsize(m, (10, 10, 3, 1)) == (10, 10, 19, 1)

  m = Parallel((mx, x) -> cat(mx, x; dims = 3), Conv((3, 3), 3 => 16; pad = 1), identity)
  @test outputsize(m, (10, 10, 3, 1)) == (10, 10, 19, 1)
end

@testset "multiple inputs" begin
  m = Parallel(vcat, Dense(2, 4, relu), Dense(3, 6, relu))
  @test outputsize(m, (2,), (3,)) == (10,)
  @test outputsize(m, ((2,), (3,))) == (10,)
  @test outputsize(m, (2,), (3,); padbatch=true) == (10, 1)
  @test outputsize(m, (2,7), (3,7)) == (10, 7)

  m = Chain(m, Dense(10, 13, tanh), softmax)
  @test outputsize(m, (2,), (3,)) == (13,)
  @test outputsize(m, ((2,), (3,))) == (13,)
  @test outputsize(m, (2,), (3,); padbatch=true) == (13, 1)
  @test outputsize(m, (2,7), (3,7)) == (13, 7)
end

@testset "activations" begin
  @testset for f in [celu, elu, gelu, hardsigmoid, hardtanh,
                     leakyrelu, lisht, logcosh, logσ, mish,
                     relu, relu6, rrelu, selu, σ, softplus,
                     softshrink, softsign, swish, tanhshrink, trelu]
    @test outputsize(Dense(10, 5, f), (10, 1)) == (5, 1)
  end
end

@testset "conv" begin
  m = Conv((3, 3), 3 => 16)
  @test outputsize(m, (10, 10, 3, 1)) == (8, 8, 16, 1)
  m = Conv((3, 3), 3 => 16; stride = 2)
  @test outputsize(m, (5, 5, 3, 1)) == (2, 2, 16, 1)
  m = Conv((3, 3), 3 => 16; stride = 2, pad = 3)
  @test outputsize(m, (5, 5, 3, 1)) == (5, 5, 16, 1)
  m = Conv((3, 3), 3 => 16; stride = 2, pad = 3, dilation = 2)
  @test outputsize(m, (5, 5, 3, 1)) == (4, 4, 16, 1)
  @test_throws DimensionMismatch outputsize(m, (5, 5, 2))
  @test outputsize(m, (5, 5, 3, 100)) == (4, 4, 16, 100)

  m = ConvTranspose((3, 3), 3 => 16)
  @test outputsize(m, (8, 8, 3, 1)) == (10, 10, 16, 1)
  m = ConvTranspose((3, 3), 3 => 16; stride = 2)
  @test outputsize(m, (2, 2, 3, 1)) == (5, 5, 16, 1)
  m = ConvTranspose((3, 3), 3 => 16; stride = 2, pad = 3)
  @test outputsize(m, (5, 5, 3, 1)) == (5, 5, 16, 1)
  m = ConvTranspose((3, 3), 3 => 16; stride = 2, pad = 3, dilation = 2)
  @test outputsize(m, (4, 4, 3, 1)) == (5, 5, 16, 1)

  m = DepthwiseConv((3, 3), 3 => 6)
  @test outputsize(m, (10, 10, 3, 1)) == (8, 8, 6, 1)
  m = DepthwiseConv((3, 3), 3 => 6; stride = 2)
  @test outputsize(m, (5, 5, 3, 1)) == (2, 2, 6, 1)
  m = DepthwiseConv((3, 3), 3 => 6; stride = 2, pad = 3)
  @test outputsize(m, (5, 5, 3, 1)) == (5, 5, 6, 1)
  m = DepthwiseConv((3, 3), 3 => 6; stride = 2, pad = 3, dilation = 2)
  @test outputsize(m, (5, 5, 3, 1)) == (4, 4, 6, 1)

  m = CrossCor((3, 3), 3 => 16)
  @test outputsize(m, (10, 10, 3, 1)) == (8, 8, 16, 1)
  m = CrossCor((3, 3), 3 => 16; stride = 2)
  @test outputsize(m, (5, 5, 3, 1)) == (2, 2, 16, 1)
  m = CrossCor((3, 3), 3 => 16; stride = 2, pad = 3)
  @test outputsize(m, (5, 5, 3, 1)) == (5, 5, 16, 1)
  m = CrossCor((3, 3), 3 => 16; stride = 2, pad = 3, dilation = 2)
  @test outputsize(m, (5, 5, 3, 1)) == (4, 4, 16, 1)

  m = AdaptiveMaxPool((2, 2))
  @test outputsize(m, (10, 10, 3, 1)) == (2, 2, 3, 1)

  m = AdaptiveMeanPool((2, 2))
  @test outputsize(m, (10, 10, 3, 1)) == (2, 2, 3, 1)

  m = GlobalMaxPool()
  @test outputsize(m, (10, 10, 3, 1)) == (1, 1, 3, 1)

  m = GlobalMeanPool()
  @test outputsize(m, (10, 10, 3, 1)) == (1, 1, 3, 1)

  m = MaxPool((2, 2))
  @test outputsize(m, (10, 10, 3, 1)) == (5, 5, 3, 1)
  m = MaxPool((2, 2); stride = 1)
  @test outputsize(m, (5, 5, 4, 1)) == (4, 4, 4, 1)
  m = MaxPool((2, 2); stride = 2, pad = 3)
  @test outputsize(m, (5, 5, 2, 1)) == (5, 5, 2, 1)

  m = MeanPool((2, 2))
  @test outputsize(m, (10, 10, 3, 1)) == (5, 5, 3, 1)
  m = MeanPool((2, 2); stride = 1)
  @test outputsize(m, (5, 5, 4, 1)) == (4, 4, 4, 1)
  m = MeanPool((2, 2); stride = 2, pad = 3)
  @test outputsize(m, (5, 5, 2, 1)) == (5, 5, 2, 1)
end

@testset "normalisation" begin
  m = Dropout(0.1)
  @test outputsize(m, (10, 10)) == (10, 10)
  @test outputsize(m, (10,); padbatch=true) == (10, 1)

  m = AlphaDropout(0.1)
  @test outputsize(m, (10, 10)) == (10, 10)
  @test outputsize(m, (10,); padbatch=true) == (10, 1)

  m = LayerNorm(32)
  @test outputsize(m, (32, 32, 3, 16)) == (32, 32, 3, 16)
  @test outputsize(m, (32, 32, 3); padbatch=true) == (32, 32, 3, 1)

  m = BatchNorm(3)
  @test outputsize(m, (32, 32, 3, 16)) == (32, 32, 3, 16)
  @test outputsize(m, (32, 32, 3); padbatch=true) == (32, 32, 3, 1)

  m = InstanceNorm(3)
  @test outputsize(m, (32, 32, 3, 16)) == (32, 32, 3, 16)
  @test outputsize(m, (32, 32, 3); padbatch=true) == (32, 32, 3, 1)

  m = GroupNorm(16, 4)
  @test outputsize(m, (32, 32, 16, 16)) == (32, 32, 16, 16)
  @test outputsize(m, (32, 32, 16); padbatch=true) == (32, 32, 16, 1)
end
