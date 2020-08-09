@testset "basic" begin
  m = Chain(Conv((3, 3), 3 => 16), Conv((3, 3), 16 => 32))
  @test outdims(m, (10, 10, 3)) == (6, 6, 32)
  @test outdims(m, (10, 10, 3, 2)) == (6, 6, 32, 2)

  m = Dense(10, 5)
  @test_throws DimensionMismatch outdims(m, (5, 2)) == (5,)
  @test outdims(m, (10,)) == (5,)

  m = Chain(Dense(10, 8, σ), Dense(8, 5), Dense(5, 2))
  @test outdims(m, (10,)) == (2,)
  @test outdims(m, (10, 30)) == (2, 30)

  m = Chain(Dense(10, 8, σ), Dense(8, 4), Dense(5, 2))
  @test_throws DimensionMismatch outdims(m, (10,))

  m = Flux.Diagonal(10)
  @test outdims(m, (10,)) == (10,)

  m = Maxout(() -> Conv((3, 3), 3 => 16), 2)
  @test outdims(m, (10, 10, 3)) == (8, 8, 16)

  m = flatten
  @test outdims(m, (5, 5, 3, 10)) == (75, 10)

  m = Chain(Conv((3, 3), 3 => 16), flatten, Dense(1024, 10))
  @test outdims(m, (10, 10, 3, 50)) == (10, 50)
end

@testset "conv" begin
  m = Conv((3, 3), 3 => 16)
  @test outdims(m, (10, 10, 3)) == (8, 8, 16)
  m = Conv((3, 3), 3 => 16; stride = 2)
  @test outdims(m, (5, 5, 3)) == (2, 2, 16)
  m = Conv((3, 3), 3 => 16; stride = 2, pad = 3)
  @test outdims(m, (5, 5, 3)) == (5, 5, 16)
  m = Conv((3, 3), 3 => 16; stride = 2, pad = 3, dilation = 2)
  @test outdims(m, (5, 5, 3)) == (4, 4, 16)
  @test_throws DimensionMismatch outdims(m, (5, 5, 2))
  @test outdims(m, (5, 5, 3, 100)) == (4, 4, 16, 100)

  m = ConvTranspose((3, 3), 3 => 16)
  @test outdims(m, (8, 8, 3)) == (10, 10, 16)
  m = ConvTranspose((3, 3), 3 => 16; stride = 2)
  @test outdims(m, (2, 2, 3)) == (5, 5, 16)
  m = ConvTranspose((3, 3), 3 => 16; stride = 2, pad = 3)
  @test outdims(m, (5, 5, 3)) == (5, 5, 16)
  m = ConvTranspose((3, 3), 3 => 16; stride = 2, pad = 3, dilation = 2)
  @test outdims(m, (4, 4, 3)) == (5, 5, 16)

  m = DepthwiseConv((3, 3), 3 => 6)
  @test outdims(m, (10, 10, 3)) == (8, 8, 6)
  m = DepthwiseConv((3, 3), 3 => 6; stride = 2)
  @test outdims(m, (5, 5, 3)) == (2, 2, 6)
  m = DepthwiseConv((3, 3), 3 => 6; stride = 2, pad = 3)
  @test outdims(m, (5, 5, 3)) == (5, 5, 6)
  m = DepthwiseConv((3, 3), 3 => 6; stride = 2, pad = 3, dilation = 2)
  @test outdims(m, (5, 5, 3)) == (4, 4, 6)

  m = CrossCor((3, 3), 3 => 16)
  @test outdims(m, (10, 10, 3)) == (8, 8, 16)
  m = CrossCor((3, 3), 3 => 16; stride = 2)
  @test outdims(m, (5, 5, 3)) == (2, 2, 16)
  m = CrossCor((3, 3), 3 => 16; stride = 2, pad = 3)
  @test outdims(m, (5, 5, 3)) == (5, 5, 16)
  m = CrossCor((3, 3), 3 => 16; stride = 2, pad = 3, dilation = 2)
  @test outdims(m, (5, 5, 3)) == (4, 4, 16)

  m = MaxPool((2, 2))
  @test outdims(m, (10, 10, 3)) == (5, 5, 3)
  m = MaxPool((2, 2); stride = 1)
  @test outdims(m, (5, 5, 4)) == (4, 4, 4)
  m = MaxPool((2, 2); stride = 2, pad = 3)
  @test outdims(m, (5, 5, 2)) == (5, 5, 2)

  m = MeanPool((2, 2))
  @test outdims(m, (10, 10, 3)) == (5, 5, 3)
  m = MeanPool((2, 2); stride = 1)
  @test outdims(m, (5, 5, 4)) == (4, 4, 4)
  m = MeanPool((2, 2); stride = 2, pad = 3)
  @test outdims(m, (5, 5, 2)) == (5, 5, 2)
end

@testset "normalisation" begin
  m = Dropout(0.1)
  @test outdims(m, (10, 10)) == (10, 10)
  @test outdims(m, (10,)) == (10,)

  m = AlphaDropout(0.1)
  @test outdims(m, (10, 10)) == (10, 10)
  @test outdims(m, (10,)) == (10,)

  m = LayerNorm(2)
  @test outdims(m, (32, 32, 3, 16)) == (32, 32, 3, 16)

  m = BatchNorm(3)
  @test outdims(m, (32, 32, 3, 16)) == (32, 32, 3, 16)

  m = InstanceNorm(3)
  @test outdims(m, (32, 32, 3, 16)) == (32, 32, 3, 16)

  if VERSION >= v"1.1"
    m = GroupNorm(16, 4)
    @test outdims(m, (32, 32, 3, 16)) == (32, 32, 3, 16)
  end
end