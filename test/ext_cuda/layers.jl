# Test layers and data/model movements on and off the GPU
# Add tests for layers and their gradients on the GPU
# Most of the forward passes should be fine being applied
# to bitstype objects, but this gives higher coverage for our use-cases
# Check that getting the gradients does not throw

# generic movement tests
@testset "Basic GPU Movement" begin
  @test gradient(x -> sum(gpu(x)), rand(Float32, 3, 3))[1] isa Matrix{Float32}
  @test gradient(x -> sum(cpu(x)), gpu(rand(Float32, 3, 3)))[1] isa CuMatrix{Float32}
end

const ACTIVATIONS = [identity, tanh]

function gpu_gradtest(name::String, layers::Vector, x_cpu, args...; 
    test_mode=false, 
    atol=1e-4, rtol=1e-4)
  @testset "$name GPU grad tests" begin
    for layer in layers
      @testset "$layer Layer GPU grad test" begin
        l_cpu = layer(args...)
        test_gradients(l_cpu, x_cpu; test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=nothing, 
                atol, rtol, test_mode)
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
gpu_gradtest("Embedding", embedding, [1,3,5], 5, 2)
gpu_gradtest("Embedding repeated indices", embedding, [1,3,5,3], 5, 2)
gpu_gradtest("Embedding integer index", embedding, 1, 5, 2)
gpu_gradtest("Embedding 2d index", embedding, [1 2; 3 4], 5, 2)
gpu_gradtest("Embedding OneHotVec index", embedding, OneHotVector(1, 5), 5, 2)
gpu_gradtest("Embedding OneHotMatrix index", embedding,  OneHotMatrix([1,2,3], 5), 5, 2)
gpu_gradtest("Embedding OneHotMatrix repeated indices", embedding, OneHotMatrix([1,2,2], 5), 5, 2)

# padding_idx=2: the input must contain the padding index to exercise the mask
EmbeddingPad(in, out) = Flux.Embedding(in => out; padding_idx=2)
embedding_pad = [EmbeddingPad]
gpu_gradtest("Embedding padding_idx", embedding_pad, [1,2,5,2], 5, 2)
gpu_gradtest("Embedding padding_idx OneHotMatrix", embedding_pad, OneHotMatrix([1,2,2,3], 5), 5, 2)

@testset "function layers" begin
  x = rand(Float32, 3, 3)
  test_gradients(x -> sum(Flux.normalise(x; dims=1)), x, test_gpu=true, test_cpu=false, 
    reference=AutoZygote(), compare=nothing)
  test_gradients(x -> sum(Flux.normalise(x; dims=2)), x, test_gpu=true, test_cpu=false, 
    reference=AutoZygote(), compare=nothing)
  test_gradients(x -> sum(Flux.normalise(x)), x, test_gpu=true, test_cpu=false, 
    reference=AutoZygote(), compare=nothing)
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
    test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=nothing)
end

@testset "Two-streams Bilinear" begin
  x = zeros(Float32,10,9) |> gpu
  y = zeros(Float32,2,9) |> gpu
  b = Flux.Bilinear(10, 2, 3) |> gpu
  @test size(b(x, y)) == (3,9)
  @test sum(abs2, b(x, y)) ≈ 0f0
  test_gradients(b |> cpu, x |> cpu, y |> cpu, 
    test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=nothing)
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
      test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=nothing)
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

@testset "Misc. BFloat16" begin
  # These tests are very far from exhaustive!
  # Comparisons use a loose `rtol`: BFloat16 keeps only 8 mantissa bits, and GPU
  # kernels often accumulate in Float32, so a pure-bf16 and a Float32 result differ.
  #
  # Normalization layers are exercised on the GPU only, against a Float32 GPU
  # reference: their generic CPU path rounds Float32->BFloat16, which can hang LLVM
  # codegen (JuliaMath/BFloat16s.jl#107).

  x = bf16(randn(Float32, 3, 4))
  gx = gpu(x)

  # Dense
  m1 = bf16(Dense(3 => 4, tanh))
  gm1 = gpu(m1)

  y1, back1 = Zygote.pullback(|>, x, m1)
  gy1, gback1 = Zygote.pullback(|>, gx, gm1)

  @test y1 ≈ m1(x)
  @test y1 ≈ cpu(gy1)  rtol=0.1
  @test eltype(y1) == eltype(m1(x)) == eltype(gy1) == BFloat16

  @test back1(one.(y1))[2].weight ≈ cpu(gback1(one.(gy1))[2].weight)  rtol=0.1
  @test eltype(gback1(one.(gy1))[2].bias) == BFloat16

  # LayerNorm converts fully to bf16 (it wraps NNlib.normalise, which has no scale/bias
  # and so no Float32-parameter requirement).
  gm2 = Chain(LayerNorm(3), Dropout(0.1)) |> bf16 |> gpu
  @test eltype(gm2(gx)) == BFloat16
  @test Float32.(gm2(gx)) ≈ f32(gm2)(f32(gx))  rtol=0.1

  # BatchNorm, InstanceNorm and GroupNorm are converted in mixed precision: statistics
  # and affine parameters stay Float32 while the data flows in bf16, so they dispatch to
  # NNlib's (cuDNN) half-precision kernels.
  gx4 = gpu(bf16(randn(Float32, 4, 4, 3, 2)))
  @testset "$(nameof(typeof(l)))" for l in (BatchNorm(3),
                                            InstanceNorm(3; affine=true, track_stats=true),
                                            GroupNorm(3, 3))
    gm = bf16(l) |> gpu
    @test eltype(gm.γ) == eltype(gm.β) == Float32          # affine params kept in Float32
    y = gm(gx4)
    @test eltype(y) == BFloat16                             # data flow stays bf16
    @test Float32.(y) ≈ f32(gm)(f32(gx4))  rtol=0.1        # matches the Float32 reference
    dγ = gradient(m -> sum(abs2, m(gx4)), gm)[1].γ
    @test eltype(dγ) == Float32                             # parameter grads follow the params
  end

  # Conv and pooling (cuDNN with NNlib ≥ 0.9.40 adds BFloat16 to its `CUDNNFloat` wrapper)
  m4 = Conv((3,), 2=>1, sigmoid, pad=1, stride=2) |> bf16
  x4 = bf16(randn(Float32, 7, 2, 1))
  @test m4(x4) ≈ cpu(gpu(m4)(gpu(x4)))  rtol=0.1
  @test eltype(gpu(m4)(gpu(x4))) == BFloat16
  dw4 = gradient((m,z) -> sum(abs2, m(z)), gpu(m4), gpu(x4))[1].weight
  @test eltype(dw4) == BFloat16
  xp = gpu(bf16(randn(Float32, 6, 4, 1)))
  for pool in [MaxPool((2,)), MeanPool((2,))]
    @test eltype(pool(xp)) == BFloat16
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
    test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=nothing)
end
