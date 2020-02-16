# Test layers and data/model movements on and off the GPU
# Add tests for layers and their gradients on the GPU
# Most of the forward passes should be fine being applied
# to bitstype objects, but this gives higher coverage for our use-cases
# Check that getting the gradients does not throw

# generic movement tests
@test_broken gradient(x -> sum(gpu(x)), rand(3,3)) isa Tuple
@test_throws ErrorException gradient(x -> sum(cpu(x)), gpu(rand(3,3))) isa Tuple

function gradtest(name::String, layers::Vector, xs = nothing, args...)
  isnothing(xs) && error("Missing input to test the layers against.")
  @testset "$name GPU grad tests" begin
    for layer in layers
      @testset "$layer GPU grad test" begin
        l = gpu(layer(args...))
        xs = gpu(xs)
        if l isa DepthwiseConv || l isa AlphaDropout
          ps = Flux.params(l)
          @test_broken gradient(() -> sum(l(xs)), ps) isa Flux.Zygote.Grads
        else
          ps = Flux.params(l)
          @test gradient(() -> sum(l(xs)), ps) isa Flux.Zygote.Grads
          gs = gradient(() -> sum(l(xs)), ps)

          # Handle pooling layers
          if !isempty(ps)
            @test gs[first(ps)] isa Flux.CuArrays.CuArray
          end
        end
      end
    end
  end
end

# Repeats from Conv, CrossCor

r = rand(Float32, 28, 28, 1, 1)
conv_layers = [Conv, ConvTranspose, CrossCor, DepthwiseConv]
gradtest("Conv", conv_layers, r, (2,2), 1=>3)

pooling_layers = [MaxPool, MeanPool]
gradtest("Pooling", pooling_layers, r, (2,2))

dropout_layers = [Dropout, AlphaDropout]
gradtest("Dropout", dropout_layers, r, 0.5f0)

norm_layers = [LayerNorm, BatchNorm]
gradtest("Normalising" norm_layers, rand(Float32, 28,28,3,1), 1)

instancenorm = [InstanceNorm]
gradtest(instancenorm, r, "InstanceNorm", 1)

groupnorm = [GroupNorm]
gradtest("GroupNorm", groupnorm, rand(Float32, 28,28,3,1), 3, 1)

const stateless_layers = [Flux.mse,
                          Flux.crossentropy,
                          Flux.logitcrossentropy,
                          Flux.normalise]

const stateless_layers_broadcasted = [Flux.binarycrossentropy,
                                      Flux.logitbinarycrossentropy]

function stateless_gradtest(f, args...)
  @test gradient((args...) -> sum(f(args...)), args...)[1] isa CuArray
end

function stateless_gradtest_broadcasted(f, args...)
  if f == Flux.binarycrossentropy
    @test_broken gradient((args...) -> sum(f.(args...)), args...)[1] isa CuArray
  else
    @test gradient((args...) -> sum(f.(args...)), args...)[1] isa CuArray
  end
end

@testset "Stateless GPU grad tests" begin
  x = gpu(rand(3,3))
  y = gpu(rand(3,3))

  for layer in stateless_layers
    if layer == Flux.normalise
      stateless_gradtest(layer, x)
    else
      stateless_gradtest(layer, x, y)
    end
  end

  for layer in stateless_layers_broadcasted
    stateless_gradtest_broadcasted(layer, x, y)
  end
end
