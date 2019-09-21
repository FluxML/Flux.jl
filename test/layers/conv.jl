using Flux, Test
using Flux: maxpool, meanpool

@testset "Pooling" begin
  x = randn(Float32, 10, 10, 3, 2)
  mp = MaxPool((2, 2))
  @test mp(x) == maxpool(x, PoolDims(x, 2))
  mp = MeanPool((2, 2))
  @test mp(x) == meanpool(x, PoolDims(x, 2))
end

@testset "CNN" begin
  r = zeros(Float32, 28, 28, 1, 5)
  m = Chain(
    Conv((2, 2), 1=>16, relu),
    MaxPool((2,2)),
    Conv((2, 2), 16=>8, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10), softmax)

  @test size(m(r)) == (10, 5)
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

  m3 = DepthwiseConv((2, 3), 3=>9)
  @test size(m3(r), 3) == 9

  # Test that we cannot ask for non-integer multiplication factors
  @test_throws AssertionError DepthwiseConv((2,2), 3=>10)
end

@testset "ConvTranspose" begin
  x = zeros(Float32, 28, 28, 1, 1)
  y = Conv((3,3), 1 => 1)(x)
  x_hat = ConvTranspose((3, 3), 1 => 1)(y)
  @test size(x_hat) == size(x)
end

@testset "CrossCor" begin
  x = rand(Float32, 28, 28, 1, 1)
  w = rand(2,2,1,1)
  y = CrossCor(w, [0.0])

  @test sum(w .* x[1:2, 1:2, :, :]) == y(x)[1, 1, 1, 1]

  r = zeros(Float32, 28, 28, 1, 5)
  m = Chain(
    CrossCor((2, 2), 1=>16, relu),
    MaxPool((2,2)),
    CrossCor((2, 2), 16=>8, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10), softmax)

  @test size(m(r)) == (10, 5)
  @test y(x) != Conv(w, [0.0])(x)
  @test CrossCor(w[end:-1:1, end:-1:1, :, :], [0.0])(x) == Conv(w, [0.0])(x)
end

@testset "Conv with non quadratic window #700" begin
  data = zeros(Float32, 7,7,1,1)
  data[4,4,1,1] = 1

  l = Conv((3,3), 1=>1)
  expected = zeros(eltype(l.weight),5,5,1,1)
  expected[2:end-1,2:end-1,1,1] = l.weight
  @test expected == l(data)

  l = Conv((3,1), 1=>1)
  expected = zeros(eltype(l.weight),5,7,1,1)
  expected[2:end-1,4,1,1] = l.weight
  @test expected == l(data)

  l = Conv((1,3), 1=>1)
  expected = zeros(eltype(l.weight),7,5,1,1)
  expected[4,2:end-1,1,1] = l.weight
  @test expected == l(data)

  @test begin
    # we test that the next expression does not throw
    randn(Float32, 10,10,1,1) |> Conv((6,1), 1=>1, Flux.σ)
    true
  end
end
