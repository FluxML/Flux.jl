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
  m.weight.data[:] .= 1.0
  m.bias.data[:] .= 0.0
  y_hat = Flux.data(m(r))[:,:,1,1]
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
  m1 = DepthwiseConv((2, 2), 3=>5)
  @test size(m1(r), 3) == 15
  m2 = DepthwiseConv((2, 2), 3)
  @test size(m2(r), 3) == 3
  
  x = zeros(Float64, 28, 28, 3, 5)
  
  m3 = DepthwiseConv((2, 2), 3 => 5)
  
  @test size(m3(r), 3) == 15
  
  m4 = DepthwiseConv((2, 2), 3)
  
  @test size(m4(r), 3) == 3
end

@testset "ConvTranspose" begin
  x = zeros(Float32, 28, 28, 1, 1)
  y = Conv((3,3), 1 => 1)(x)
  x_hat = ConvTranspose((3, 3), 1 => 1)(y)
  @test size(x_hat) == size(x)
end
