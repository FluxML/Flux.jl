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

  m = Flux.Scale(10)
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
  m2 = LayerNorm(3, 2)
  @test outputsize(m2, (3, 2)) == (3, 2) == size(m2(randn(3, 2)))
  @test outputsize(m2, (3,)) == (3, 2) == size(m2(randn(3, 2)))

  m = BatchNorm(3)
  @test outputsize(m, (32, 32, 3, 16)) == (32, 32, 3, 16)
  @test outputsize(m, (32, 32, 3); padbatch=true) == (32, 32, 3, 1)
  @test_throws Exception m(randn(Float32, 32, 32, 5, 1))
  @test_throws DimensionMismatch outputsize(m, (32, 32, 5, 1))

  m = InstanceNorm(3)
  @test outputsize(m, (32, 32, 3, 16)) == (32, 32, 3, 16)
  @test outputsize(m, (32, 32, 3); padbatch=true) == (32, 32, 3, 1)
  @test_throws Exception m(randn(Float32, 32, 32, 5, 1))
  @test_throws DimensionMismatch outputsize(m, (32, 32, 5, 1))

  m = GroupNorm(16, 4)
  @test outputsize(m, (32, 32, 16, 16)) == (32, 32, 16, 16)
  @test outputsize(m, (32, 32, 16); padbatch=true) == (32, 32, 16, 1)
  @test_throws Exception m(randn(Float32, 32, 32, 15, 4))
  @test_throws DimensionMismatch outputsize(m, (32, 32, 15, 4))
end

@testset "autosize macro" begin
  m = @autosize (3,) Dense(_ => 4)
  @test randn(3) |> m |> size == (4,)

  m = @autosize (3, 1) Chain(Dense(_ => 4), Dense(4 => 10), softmax)
  @test randn(3, 5) |> m |> size == (10, 5)
  
  m = @autosize (2, 3, 4, 5) Dense(_ => 10)  # goes by first dim, not 2nd-last
  @test randn(2, 3, 4, 5) |> m |> size == (10, 3, 4, 5)
  
  m = @autosize (9,) Dense(_ => div(_,2))
  @test randn(9) |> m |> size == (4,)

  m = @autosize (3,) Chain(one = Dense(_ => 4), two = softmax)  # needs kw
  @test randn(3) |> m |> size == (4,)

  m = @autosize (3, 45) Maxout(() -> Dense(_ => 6, tanh), 2)    # needs ->, block
  @test randn(3, 45) |> m |> size == (6, 45)

  # here Parallel gets two inputs, no problem:
  m = @autosize (3,) Chain(SkipConnection(Dense(_ => 4), Parallel(vcat, Dense(_ => 5), Dense(_ => 6))), Flux.Scale(_))
  @test randn(3) |> m |> size == (11,)
  
  # like Dense, LayerNorm goes by the first dimension:
  m = @autosize (3, 4, 5) LayerNorm(_)
  @test rand(3, 6, 7) |> m |> size == (3, 6, 7)

  m = @autosize (3, 3, 10) LayerNorm(_, _)  # does not check that sizes match
  @test rand(3, 3, 10) |> m |> size == (3, 3, 10)
  
  m = @autosize (3,) Flux.Bilinear(_ => 10)
  @test randn(3) |> m |> size == (10,)

  m = @autosize (3, 1) Flux.Bilinear(_ => 10)
  @test randn(3, 4) |> m |> size == (10, 4)
  
  @test_throws Exception @eval @autosize (3,) Flux.Bilinear((_,3) => 10)
  
  # first docstring example
  m = @autosize (3, 1) Chain(Dense(_ => 2, sigmoid), BatchNorm(_, affine=false))
  @test randn(3, 4) |> m |> size == (2, 4)
  
  # evil docstring example
  img = [28, 28];
  m = @autosize (img..., 1, 32) Chain(              # size is only needed at runtime
         Chain(c = Conv((3,3), _ => 5; stride=2, pad=SamePad()),
               p = MeanPool((3,3)),
               b = BatchNorm(_),
               f = Flux.flatten),
         Dense(_ => _÷4, relu, init=Flux.rand32),   # can calculate output size _÷4
         SkipConnection(Dense(_ => _, relu), +),
         Dense(_ => 10),
      ) |> gpu                                      # moves to GPU after initialisation
  @test randn(Float32, img..., 1, 32) |> gpu |> m |> size == (10, 32)
end