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

# TODO: These layers get into scalar indexing
# `AlphaDropout` throws a compilation error on GPUs,
# whereas, the rest are scalar indexing issues.
const BROKEN_LAYERS = [DepthwiseConv,
		                   AlphaDropout,
                       InstanceNorm,
                       GroupNorm]

function gradtest(name::String, layers::Vector, xs = nothing, args...)
  isnothing(xs) && error("Missing input to test the layers against.")
  @testset "$name GPU grad tests" begin
    for layer in layers
      @testset "$layer GPU grad test" begin
        l = gpu(layer(args...))
        xs = gpu(xs)
        if any(x -> isa(l, x), BROKEN_LAYERS)
          ps = Flux.params(l)
          @test_broken gradient(() -> sum(l(xs)), ps) isa Flux.Zygote.Grads
        else
          ps = Flux.params(l)
          @test gradient(() -> sum(l(xs)), ps) isa Flux.Zygote.Grads
          gs = gradient(() -> sum(l(xs)), ps)

          # Handle pooling layers
          if !isempty(ps)
            @test gs[first(ps)] isa Flux.CUDA.CuArray
          end
        end
      end
    end
  end
end

# Repeats from Conv, CrossCor

# Just to give testset in gradtest meaningful labels
ConvNoBias(args...) = Conv(args...; bias=false)
ConvTransposeNoBias(args...) = ConvTranspose(args...; bias=false)
CrossCorNoBias(args...) = CrossCor(args...; bias=false)
DepthwiseConvNoBias(args...) = DepthwiseConv(args...;bias=false)
r = rand(Float32, 28, 28, 1, 1)
conv_layers = [Conv, ConvNoBias, ConvTranspose, ConvTransposeNoBias, CrossCor, CrossCorNoBias, DepthwiseConv, DepthwiseConvNoBias]
gradtest("Conv", conv_layers, r, (2,2), 1=>3)

pooling_layers = [MaxPool, MeanPool]
gradtest("Pooling", pooling_layers, r, (2,2))

adaptive_pooling_layers = [AdaptiveMaxPool, AdaptiveMeanPool]
gradtest("AdaptivePooling", adaptive_pooling_layers, r, (7,7))

dropout_layers = [Dropout, AlphaDropout]
gradtest("Dropout", dropout_layers, r, 0.5f0)

norm_layers = [LayerNorm, BatchNorm]
gradtest("Normalising", norm_layers, rand(Float32, 28,28,3,1), 1)

instancenorm = [InstanceNorm]
gradtest("InstanceNorm", instancenorm, r, 1)

groupnorm = [GroupNorm]
gradtest("GroupNorm", groupnorm, rand(Float32, 28,28,3,1), 3, 1)

const stateless_layers = [Flux.normalise]

const stateless_layers_broadcasted = []

function stateless_gradtest(f, args...)
  @test gradient((args...) -> sum(f(args...)), args...)[1] isa CuArray
end

function stateless_gradtest_broadcasted(f, args...)
  @test gradient((args...) -> sum(f.(args...)), args...)[1] isa CuArray
end

@testset "Stateless GPU grad tests" begin
  x = gpu(rand(3,3))
  y = gpu(rand(3,3))

  for layer in stateless_layers
    if layer == Flux.normalise
      stateless_gradtest(x -> layer(x, dims=1), x)
    else
      stateless_gradtest(layer, x, y)
    end
  end

  for layer in stateless_layers_broadcasted
    stateless_gradtest_broadcasted(layer, x, y)
  end
end

@testset "Zeros mapped for $cl" for cl in (Conv, ConvTranspose, CrossCor, DepthwiseConv)
  l = cl((2,2), 1=>3, bias = false) |> gpu
  ip = zeros(Float32, 28,28,1,1) |> gpu
  if cl in BROKEN_LAYERS
    @test_broken sum(l(ip)) ≈ 0.f0
    @test_broken gradient(() -> sum(l(ip)), Flux.params(l)) isa Flux.Zygote.Grads
  else
    @test sum(l(ip)) ≈ 0.f0
    gs = gradient(() -> sum(l(ip)), Flux.params(l))
    @test l.bias ∉ gs.params 
  end
end

@testset "Dense with Zeros bias" begin
  l = Dense(ones(Float32, 4,3), Flux.Zeros()) |> gpu
  ip = zeros(Float32, 3, 7) |> gpu

  @test sum(l(ip)) ≈ 0.f0  
  gs = gradient(() -> sum(l(ip)), Flux.params(l))
  @test l.b ∉ gs.params 
end