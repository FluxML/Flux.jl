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

# TODO: These layers get into scalar indexing issues.
const BROKEN_LAYERS = Union{}

const ACTIVATIONS = [identity, relu, tanh,
                     sigmoid, exp, softplus,
                     elu, selu]

function gpu_gradtest(name::String, layers::Vector, x_cpu = nothing, args...; test_cpu = true, test_mode = false)
  isnothing(x_cpu) && error("Missing input to test the layers against.")
  @testset "$name GPU grad tests" begin
    for layer in layers
      @testset "$layer Layer GPU grad test" begin

        # compute output and grad of parameters
        l_cpu = layer(args...)
        l_gpu = l_cpu |> gpu
        if test_mode
          testmode!(l_cpu)
          testmode!(l_gpu)
        end

        ps_cpu = Flux.params(l_cpu)
        y_cpu, back_cpu = pullback(() -> sum(l_cpu(x_cpu)), ps_cpu)
        gs_cpu = back_cpu(1f0)

        x_gpu = gpu(x_cpu)
        ps_gpu = Flux.params(l_gpu)

        if typeof(l_gpu) <: BROKEN_LAYERS
          @test_broken gradient(() -> sum(l_gpu(x_gpu)), ps_gpu) isa Flux.Zygote.Grads
        else
          y_gpu, back_gpu = pullback(() -> sum(l_gpu(x_gpu)), ps_gpu)
          gs_gpu = back_gpu(1f0) # TODO many layers error out when backprop int 1, should fix

          # compute grad of input
          xg_cpu = gradient(x -> sum(l_cpu(x)), x_cpu)[1]
          xg_gpu = gradient(x -> sum(l_gpu(x)), x_gpu)[1]

          # test
          if test_cpu
            if layer === GroupedConvTranspose
              @test y_gpu ≈ y_cpu rtol=1f-2 atol=1f-3
            else
              @test y_gpu ≈ y_cpu rtol=1f-3 atol=1f-3
            end
            if isnothing(xg_cpu)
              @test isnothing(xg_gpu)
            else
              if layer === GroupedConvTranspose
                @test Array(xg_gpu) ≈ xg_cpu rtol = 2f-2 atol = 1f-3
              else
                @test Array(xg_gpu) ≈ xg_cpu rtol = 1f-3 atol = 1f-3
              end
            end
          end
          @test gs_gpu isa Flux.Zygote.Grads
          for (p_cpu, p_gpu) in zip(ps_cpu, ps_gpu)
            if isnothing(gs_cpu[p_cpu])
              @test isnothing(gs_gpu[p_gpu])
            else
              @test gs_gpu[p_gpu] isa CuArray
              if test_cpu
                @test Array(gs_gpu[p_gpu]) ≈ gs_cpu[p_cpu] rtol=1f-3 atol=1f-3
              end
            end
          end
        end
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
  gpu_gradtest("Convolution with $act", conv_layers, r, (2,2), 1=>3, act, test_cpu = false)

  groupedconv = [GroupedConv, GroupedConvTranspose]
  gpu_gradtest("GroupedConvolution with $act", groupedconv, rand(Float32, 28, 28, 100, 2), (3,3), 100 => 25, act, test_cpu = true)

  batch_norm = [BatchNorm, BatchNormNoTrackStats]
  gpu_gradtest("BatchNorm 1 with $act", batch_norm, rand(Float32, 28,28,3,4), 3, act, test_cpu = false) #TODO fix errors
  gpu_gradtest("BatchNorm 2 with $act", batch_norm, rand(Float32, 5,4), 5, act, test_cpu = true)

  batch_norm = [BatchNormNoTrackStats]
  gpu_gradtest("BatchNorm 3 with $act (test mode)", batch_norm, rand(Float32, 5,4), 5, act, test_cpu = true, test_mode = true)

  instancenorm = [InstanceNorm]
  gpu_gradtest("InstanceNorm with $act", instancenorm, r, 1, act, test_cpu = false)

  groupnorm = [GroupNorm]
  gpu_gradtest("GroupNorm with $act", groupnorm, rand(Float32, 28,28,3,1), 3, 1, act, test_cpu = false)
end

r = rand(Float32, 28, 28, 1, 1)

pooling_layers = [MaxPool, MeanPool]
gpu_gradtest("Pooling", pooling_layers, r, (2,2))

adaptive_pooling_layers = [AdaptiveMaxPool, AdaptiveMeanPool]
gpu_gradtest("AdaptivePooling", adaptive_pooling_layers, r, (7,7), test_cpu = false)

dropout_layers = [Dropout, AlphaDropout]
gpu_gradtest("Dropout", dropout_layers, r, 0.5f0; test_cpu = false) # dropout is not deterministic

layer_norm = [LayerNorm]
gpu_gradtest("LayerNorm 1", layer_norm, rand(Float32, 28,28,3,4), 28, test_cpu = false) #TODO fix errors
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

@testset "function layers" begin
  x = rand(Float32, 3,3)
  gpu_autodiff_test(x -> sum(Flux.normalise(x; dims=1)), x)
  gpu_autodiff_test(x -> sum(Flux.normalise(x; dims=2)), x)
  gpu_autodiff_test(x -> sum(Flux.normalise(x)), x)
end

@testset "Zeros mapped for $cl" for cl in (Conv, ConvTranspose, CrossCor, DepthwiseConv)
  l = cl((2,2), 1=>3, bias = false) |> gpu
  ip = zeros(Float32, 28,28,1,1) |> gpu
  if typeof(l) <: BROKEN_LAYERS
    @test_broken sum(l(ip)) ≈ 0.f0
    @test_broken gradient(() -> sum(l(ip)), Flux.params(l)) isa Flux.Zygote.Grads
  else
    @test sum(l(ip)) ≈ 0.f0
    gs = gradient(() -> sum(l(ip)), Flux.params(l))
    @test l.bias ∉ gs.params
  end
end

@testset "Dense without bias" begin
  l = Dense(ones(Float32, 4, 3), false) |> gpu
  ip = zeros(Float32, 3, 7) |> gpu

  @test sum(l(ip)) ≈ 0.f0
  gs = gradient(() -> sum(l(ip)), Flux.params(l))
  @test l.bias ∉ gs.params
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
  gradient(() -> sum(m_cpu(x_cpu)), Flux.params(m_cpu))
  @test !(m_cpu.μ ≈ μ_cpu)

  μ_gpu = copy(m_gpu.μ)
  m_gpu(x_gpu)
  @test m_gpu.μ ≈ μ_gpu
  gradient(() -> sum(m_gpu(x_gpu)), Flux.params(m_gpu))
  @test !(m_gpu.μ ≈ μ_gpu)

  @test Array(m_gpu.μ) ≈ m_cpu.μ

  ## In testmode, never track statistics
  testmode!(m_cpu)
  μ_cpu = copy(m_cpu.μ)
  m_cpu(x_cpu)
  @test m_cpu.μ ≈ μ_cpu
  gradient(() -> sum(m_cpu(x_cpu)), Flux.params(m_cpu))
  @test m_cpu.μ ≈ μ_cpu

  testmode!(m_gpu)
  μ_gpu = copy(m_gpu.μ)
  m_gpu(x_gpu)
  @test m_gpu.μ ≈ μ_gpu
  gradient(() -> sum(m_gpu(x_gpu)), Flux.params(m_gpu))
  @test m_gpu.μ ≈ μ_gpu

  ## In trainmode, always track statistics
  trainmode!(m_cpu)
  μ_cpu = copy(m_cpu.μ)
  m_cpu(x_cpu)
  @test !(m_cpu.μ ≈ μ_cpu)
  μ_cpu = copy(m_cpu.μ)
  gradient(() -> sum(m_cpu(x_cpu)), Flux.params(m_cpu))
  @test !(m_cpu.μ ≈ μ_cpu)

  trainmode!(m_gpu)
  μ_gpu = copy(m_gpu.μ)
  m_gpu(x_gpu)
  @test !(m_gpu.μ ≈ μ_gpu)
  μ_gpu = copy(m_gpu.μ)
  gradient(() -> sum(m_gpu(x_gpu)), Flux.params(m_gpu))
  @test !(m_gpu.μ ≈ μ_gpu)

  ## No errors if input type mistmatch
  # x_cpu = rand(Float64, 3, 2, 2)
  # x_gpu = x_cpu |> gpu
  # m_cpu(x_cpu)
  # gradient(() -> sum(m_cpu(x_cpu)), Flux.params(m_cpu))
  # m_gpu(x_gpu)
  # gradient(() -> sum(m_gpu(x_gpu)), Flux.params(m_gpu))
end

@testset "Two-streams Bilinear" begin
  x = zeros(Float32,10,9) |> gpu
  y = zeros(Float32,2,9) |> gpu
  b = Flux.Bilinear(10, 2, 3) |> gpu
  @test size(b(x,y)) == (3,9)
  @test sum(abs2, b(x,y)) ≈ 0f0
  gs_gpu = gradient(() -> sum(abs2.(b(x, y))), params(b))
  b_cpu, x_cpu, y_cpu = b |> cpu, x |> cpu, y |> cpu
  gs_cpu = gradient(() -> sum(abs2.(b_cpu(x_cpu, y_cpu))), params(b_cpu))
  for (pgpu, pcpu) in zip(params(b), params(b_cpu))
    @test gs_cpu[pcpu] ≈ Array(gs_gpu[pgpu])
  end
end

@testset "Two-streams Bilinear" begin
  x = zeros(Float32,10,9) |> gpu
  y = zeros(Float32,2,9) |> gpu
  b = Flux.Bilinear(10, 2, 3) |> gpu
  @test size(b(x,y)) == (3,9)
  @test sum(abs2, b(x,y)) ≈ 0f0
  gs_gpu = gradient(() -> sum(abs2.(b(x, y))), params(b))
  b_cpu, x_cpu, y_cpu = b |> cpu, x |> cpu, y |> cpu
  gs_cpu = gradient(() -> sum(abs2.(b_cpu(x_cpu, y_cpu))), params(b_cpu))
  for (pgpu, pcpu) in zip(params(b), params(b_cpu))
    @test gs_cpu[pcpu] ≈ Array(gs_gpu[pgpu])
  end
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
    input_cpu = randn(10, 10, 10, 10)
    input_gpu = input_cpu |> gpu
    layer_cpu = Parallel(+, x -> zero(x), identity)
    layer_gpu = layer_cpu |> gpu
    gs_cpu = gradient(() -> sum(abs2.(layer_cpu(input_cpu))), params(layer_cpu))
    gs_gpu = gradient(() -> sum(abs2.(layer_gpu(input_gpu))), params(layer_gpu))
    for (pgpu, pcpu) in zip(params(layer_cpu), params(layer_gpu))
      @test gs_cpu[pcpu] ≈ gs_gpu[pgpu]
    end
  end
end

@testset "Dropout RNGs" begin
  @test_throws ArgumentError Flux.dropout(MersenneTwister(), CUDA.rand(Float32, 2, 3), 0.1)
  @testset for layer in (Dropout, AlphaDropout)
    m = layer(0.1; rng = MersenneTwister(123))
    @test_throws ErrorException gpu(m)
    m = layer(0.1; rng = CUDA.default_rng())
    @test gpu(m).rng isa CUDA.RNG
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

  gm_cpu, gx_cpu = gradient(mha_cpu, x_cpu) do mha, x
    y, α = mha(x)
    return sum(y.^2) + sum(α.^2)
  end
  gm_gpu, gx_gpu = gradient(mha_gpu, x_gpu) do mha, x
    y, α = mha(x)
    return sum(y.^2) + sum(α.^2)
  end
  check_grad(gm_gpu, gm_cpu)
  check_grad(gx_gpu, gx_cpu)
end
