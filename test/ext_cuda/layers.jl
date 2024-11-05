# Test layers and data/model movements on and off the GPU
# Add tests for layers and their gradients on the GPU
# Most of the forward passes should be fine being applied
# to bitstype objects, but this gives higher coverage for our use-cases
# Check that getting the gradients does not throw

# generic movement tests
@testset "Basic GPU Movement" begin
  @test gradient(x -> sum(gpu(x)), rand(3,3)) isa Tuple
  @test gradient(x -> sum(cpu(x)), gpu(rand(3,3))) isa Tuple
end


const ACTIVATIONS = [identity, tanh]

function gpu_gradtest(name::String, layers::Vector, x_cpu, args...; 
    test_mode=false, test_grad_x=true, 
    atol=1e-4, rtol=1e-4)
  @testset "$name GPU grad tests" begin
    for layer in layers
      @testset "$layer Layer GPU grad test" begin

        # compute output and grad of parameters
        l_cpu = layer(args...)
        if test_mode
          testmode!(l_cpu)
        end

        test_gradients(l_cpu, x_cpu; test_gpu=true, compare_finite_diff=false, test_grad_x, atol, rtol)
      end
    end
  end
end

# Just to give testset in gpu_gradtest meaningful labels
BatchNormNoTrackStats(args...) = BatchNorm(args...; track_stats = false)
ConvNoBias(args...) = Conv(args...; bias = false)
ConvTransposeNoBias(args...) = ConvTranspose(args...; bias = false)
CrossCorNoBias(args...) = CrossCor(args...; bias = false)
DepthwiseConvNoBias(args...) = DepthwiseConv(args...; bias = false)
GroupedConv(args...) = Conv(args..., groups = 5)
GroupedConvTranspose(args...) = ConvTranspose(args..., groups = 5)

for act in ACTIVATIONS
  r = rand(Float32, 28, 28, 1, 1)
  conv_layers = [Conv, ConvNoBias,
                 ConvTranspose, ConvTransposeNoBias,
                 CrossCor, CrossCorNoBias,
                 DepthwiseConv, DepthwiseConvNoBias]
  gpu_gradtest("Convolution with $act", conv_layers, r, (2,2), 1=>3, act)

  groupedconv = [GroupedConv, GroupedConvTranspose]
  gpu_gradtest("GroupedConvolution with $act", groupedconv, rand(Float32, 28, 28, 100, 2), (3,3), 100 => 25, act)

  batch_norm = [BatchNorm, BatchNormNoTrackStats]
  gpu_gradtest("BatchNorm 1 with $act", batch_norm, rand(Float32, 28,28,3,4), 3, act, atol=1e-3)
  gpu_gradtest("BatchNorm 2 with $act", batch_norm, rand(Float32, 5,4), 5, act, atol=1e-3)

  batch_norm = [BatchNormNoTrackStats]
  gpu_gradtest("BatchNorm 3 with $act (test mode)", batch_norm, rand(Float32, 5,4), 5, act, 
    test_mode=true, atol=1e-3)

  instancenorm = [InstanceNorm]
  gpu_gradtest("InstanceNorm with $act", instancenorm, r, 1, act)

  groupnorm = [GroupNorm]
  gpu_gradtest("GroupNorm with $act", groupnorm, rand(Float32, 28,28,3,1), 3, 1, act)
end

r = rand(Float32, 28, 28, 1, 1)

pooling_layers = [MaxPool, MeanPool]
gpu_gradtest("Pooling", pooling_layers, r, (2,2))

adaptive_pooling_layers = [AdaptiveMaxPool, AdaptiveMeanPool]
gpu_gradtest("AdaptivePooling", adaptive_pooling_layers, r, (7,7))

dropout_layers = [Dropout, AlphaDropout]
gpu_gradtest("Dropout", dropout_layers, r, 1e-6) # dropout is not deterministic

layer_norm = [LayerNorm]
gpu_gradtest("LayerNorm 1", layer_norm, rand(Float32, 28,28,3,4), 28)
gpu_gradtest("LayerNorm 2", layer_norm, rand(Float32, 5,4), 5)

upsample = [x -> Upsample(scale=x)]
gpu_gradtest("Upsample 2d", upsample, rand(Float32, 3, 4, 2, 3), (2,2))
gpu_gradtest("Upsample 1d", upsample, rand(Float32, 3, 4, 2, 3), (2,))

pixelshuffle = [PixelShuffle]
gpu_gradtest("PixelShuffle 2d", pixelshuffle, rand(Float32, 3, 4, 18, 3), 3)
gpu_gradtest("PixelShuffle 1d", pixelshuffle, rand(Float32, 3, 18, 3), 3)

embedding = [Flux.Embedding]
gpu_gradtest("Embedding", embedding, [1,3,5], 5, 2, test_grad_x=false)
gpu_gradtest("Embedding repeated indices", embedding, [1,3,5,3], 5, 2, test_grad_x=false)
gpu_gradtest("Embedding integer index", embedding, 1, 5, 2, test_grad_x=false)
gpu_gradtest("Embedding 2d index", embedding, [1 2; 3 4], 5, 2, test_grad_x=false)
gpu_gradtest("Embedding OneHotVec index", embedding, OneHotVector(1, 5), 5, 2, test_grad_x=false)
gpu_gradtest("Embedding OneHotMatrix index", embedding,  OneHotMatrix([1,2,3], 5), 5, 2, test_grad_x=false)
gpu_gradtest("Embedding OneHotMatrix repeated indices", embedding, OneHotMatrix([1,2,2], 5), 5, 2, test_grad_x=false)

@testset "function layers" begin
  x = rand(Float32, 3, 3)
  test_gradients(x -> sum(Flux.normalise(x; dims=1)), x, test_gpu=true, compare_finite_diff=false)
  test_gradients(x -> sum(Flux.normalise(x; dims=2)), x, test_gpu=true, compare_finite_diff=false)
  test_gradients(x -> sum(Flux.normalise(x)), x, test_gpu=true, compare_finite_diff=false)
end

@testset "Zeros mapped for $cl" for cl in (Conv, ConvTranspose, CrossCor, DepthwiseConv)
  l = cl((2,2), 1=>3, bias = false) |> gpu
  ip = zeros(Float32, 28,28,1,1) |> gpu
  @test sum(l(ip)) ≈ 0.f0
  gs = gradient(l -> sum(l(ip)), l)[1]
  @test gs.bias === nothing
end

@testset "Dense without bias" begin
  l = Dense(ones(Float32, 4, 3), false) |> gpu
  ip = zeros(Float32, 3, 7) |> gpu

  @test sum(l(ip)) ≈ 0.f0
  gs = gradient(l -> sum(l(ip)), l)[1]
  @test gs.bias === nothing
end

@testset "Extended BatchNorm" begin
  m_cpu = BatchNorm(2)
  m_gpu = m_cpu |> gpu
  x_cpu = rand(Float32, 3, 2, 2)
  x_gpu = x_cpu |> gpu

  ## In :auto mode, track statistics only in gradient contest
  μ_cpu = copy(m_cpu.μ)
  m_cpu(x_cpu)
  @test m_cpu.μ ≈ μ_cpu
  gradient(m_cpu -> sum(m_cpu(x_cpu)), m_cpu)
  @test !(m_cpu.μ ≈ μ_cpu)

  μ_gpu = copy(m_gpu.μ)
  m_gpu(x_gpu)
  @test m_gpu.μ ≈ μ_gpu
  gradient(m_gpu -> sum(m_gpu(x_gpu)), m_gpu)
  @test !(m_gpu.μ ≈ μ_gpu)

  @test Array(m_gpu.μ) ≈ m_cpu.μ

  ## In testmode, never track statistics
  testmode!(m_cpu)
  μ_cpu = copy(m_cpu.μ)
  m_cpu(x_cpu)
  @test m_cpu.μ ≈ μ_cpu
  gradient(m_cpu -> sum(m_cpu(x_cpu)), m_cpu)
  @test m_cpu.μ ≈ μ_cpu

  testmode!(m_gpu)
  μ_gpu = copy(m_gpu.μ)
  m_gpu(x_gpu)
  @test m_gpu.μ ≈ μ_gpu
  gradient(m_gpu -> sum(m_gpu(x_gpu)), m_gpu)
  @test m_gpu.μ ≈ μ_gpu

  ## In trainmode, always track statistics
  trainmode!(m_cpu)
  μ_cpu = copy(m_cpu.μ)
  m_cpu(x_cpu)
  @test !(m_cpu.μ ≈ μ_cpu)
  μ_cpu = copy(m_cpu.μ)
  gradient(m_cpu -> sum(m_cpu(x_cpu)), m_cpu)
  @test !(m_cpu.μ ≈ μ_cpu)

  trainmode!(m_gpu)
  μ_gpu = copy(m_gpu.μ)
  m_gpu(x_gpu)
  @test !(m_gpu.μ ≈ μ_gpu)
  μ_gpu = copy(m_gpu.μ)
  gradient(m_gpu -> sum(m_gpu(x_gpu)), m_gpu)
  @test !(m_gpu.μ ≈ μ_gpu)
end

@testset "Two-streams Bilinear" begin
  x = zeros(Float32,10,9) |> gpu
  y = zeros(Float32,2,9) |> gpu
  b = Flux.Bilinear(10, 2, 3) |> gpu
  @test size(b(x, y)) == (3,9)
  @test sum(abs2, b(x, y)) ≈ 0f0
  test_gradients(b |> cpu, x |> cpu, y |> cpu, 
    test_gpu=true, compare_finite_diff=false, loss=(m, x, y) -> mean(abs2, m(x, y)))
end

@testset "Two-streams Bilinear" begin
  x = zeros(Float32,10,9) |> gpu
  y = zeros(Float32,2,9) |> gpu
  b = Flux.Bilinear(10, 2, 3) |> gpu
  @test size(b(x, y)) == (3,9)
  @test sum(abs2, b(x, y)) ≈ 0f0
  test_gradients(b |> cpu, x |> cpu, y |> cpu, 
    test_gpu=true, compare_finite_diff=false, loss=(m, x, y) -> mean(abs2, m(x, y)))
end

@testset "Parallel" begin
  @testset "zero sum" begin
    input = randn(10, 10, 10, 10) |> gpu
    layer_gpu = Parallel(+, zero, identity) |> gpu
    @test layer_gpu(input) == input
    @test layer_gpu(input) isa CuArray
  end

  @testset "vararg input" begin
    inputs = (randn(10), randn(5), randn(4)) .|> gpu
    layer = Parallel(+, Dense(10, 2), Dense(5, 2), Dense(4, 2)) |> gpu
    @test size(layer(inputs)) == (2,)
  end

  @testset "gradient" begin
    layer_cpu = Parallel(+, x -> zero(x), identity)
    test_gradients(layer_cpu, randn(2, 2, 2, 2), 
      test_gpu=true, compare_finite_diff=false, loss=(m, x) -> mean(abs2, m(x)))
  end
end

@testset "Dropout RNGs" begin
  @test_throws ArgumentError Flux.dropout(MersenneTwister(), CUDA.rand(Float32, 2, 3), 0.1)
  @testset for layer in (Dropout, AlphaDropout)
    m = layer(0.1)
    @test m.rng === Random.default_rng()
    @test gpu(m).rng isa CUDA.RNG
    @test cpu(gpu(m)).rng === Random.default_rng()
  end
end

@testset "Misc. Float16" begin
  # These tests are very far from exhaustive!

  x = randn(Float16, 3, 4)
  gx = gpu(x)

  # Dense
  m1 = f16(Dense(3 => 4, tanh))
  gm1 = gpu(m1)

  y1, back1 = Zygote.pullback(|>, x, m1)
  gy1, gback1 = Zygote.pullback(|>, gx, gm1)

  @test y1 ≈ m1(x) ≈ cpu(gy1)
  @test eltype(y1) == eltype(m1(x)) == eltype(gy1) == Float16

  @test back1(one.(y1))[2].weight ≈ cpu(gback1(one.(gy1))[2].weight)
  @test eltype(gback1(one.(gy1))[2].bias) == Float16

  # A fake loss with Float32
  f1(x) = sum((Float32.(x) .- 1).^2)
  @test gradient(f1, x)[1] ≈ cpu(gradient(f1, gx)[1])
  @test eltype(gradient(f1, gx)[1]) == Float16

  # Normalisation
  m2 = Chain(LayerNorm(3), Dropout(0.1)) |> f16
  gm2 = m2 |> gpu
  @test m2(x) ≈ cpu(gm2(gx))
  @test eltype(m2(x)) == Float16
  @test eltype(gm2(gx)) == Float16

  # Conv
  x3 = randn(Float16, 7, 2, 1)
  m3 = Conv((3,), 2=>1, sigmoid, pad=1, stride=2) |> f16
  @test m3(x3) ≈ f16(f32(m3)(f32(x3))) ≈ cpu(gpu(m3)(gpu(x3)))
  @test eltype(m3(x3)) == Float16
  dw = gradient((m,x) -> sum(abs2, m(x)), m3, x3)[1].weight
  @test dw ≈ f16(gradient((m,x) -> sum(abs2, m(x)), f32(m3), f32(x3))[1].weight)
  @test dw ≈ cpu(gradient((m,x) -> sum(abs2, m(x)), gpu(m3), gpu(x3))[1].weight)
  @test eltype(dw) == Float16

  # Pooling
  for pool in [MaxPool((2,)), MeanPool((2,))]
    pool(reshape(x,3,4,1)) ≈ cpu(pool(reshape(gx,3,4,1)))
    @test eltype(pool(reshape(gx,3,4,1))) == Float16
  end
end

@testset "MultiHeadAttention" begin
  dim = 4; nheads = 2; len = 3; batch_size = 5
  mha_cpu = MultiHeadAttention(dim; nheads)
  x_cpu = rand(Float32, (dim, len, batch_size))
  y_cpu, α_cpu = mha_cpu(x_cpu)

  mha_gpu = mha_cpu |> gpu
  x_gpu = x_cpu |> gpu
  y_gpu, α_gpu = mha_gpu(x_gpu)
  @test y_gpu isa CuArray{Float32}
  @test α_gpu isa CuArray{Float32}
  @test Array(y_gpu) ≈ y_cpu atol=1e-4
  @test Array(α_gpu) ≈ α_cpu atol=1e-4

  function loss(m, x)
    y, α = m(x)
    return sum(y.^2) + sum(α.^2)
  end
  test_gradients(mha_cpu, x_cpu; loss, 
    test_gpu=true, compare_finite_diff=false)
end
