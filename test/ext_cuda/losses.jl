
@testset "Losses" begin

x = [1.,2.,3.]
cx = gpu(x)
@test crossentropy(x,x) ≈ crossentropy(cx,cx)
@test crossentropy(x,x, agg=identity) ≈ crossentropy(cx,cx, agg=identity) |> cpu
@test crossentropy(x,x, agg=x->mean([1.0;2.0;3.0].*x)) ≈ crossentropy(cx,cx, agg=x->mean(gpu([1.0;2.0;3.0]).*x))

x = [-1.1491, 0.8619, 0.3127]
y = [1, 1, 0.]
@test binarycrossentropy(σ.(x), y) ≈ binarycrossentropy(gpu(σ.(x)), gpu(y))
@test logitbinarycrossentropy(x, y) ≈ logitbinarycrossentropy(gpu(x), gpu(y))

x = [0.268941  0.5  0.268941
     0.731059  0.5  0.731059]
y = [0  1  0
     1  0  1]
@test binary_focal_loss(x, y) ≈ binary_focal_loss(gpu(x), gpu(y))

x = softmax(reshape(-7:7, 3, 5) .* 1f0)
y = [1  0  0  0  1
     0  1  0  1  0
     0  0  1  0  0]
@test focal_loss(x, y) ≈ focal_loss(gpu(x), gpu(y))

@testset "GPU: $loss" for loss in ALL_LOSSES
  # let's stay far from the boundaries to avoid problems with finite differences gradients
  x = 0.1f0 .+ 0.8f0 .* rand(Float32, 3, 4)
  y = 0.1f0 .+ 0.8f0 .* rand(Float32, 3, 4)
  @test loss(x, y) ≈ loss(gpu(x), gpu(y))

  test_gradients(loss, x, y, test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=nothing)

  @testset "$lowprec" for lowprec in (f16, bf16)
    x16, y16 = lowprec(x), lowprec(y)
    l = loss(gpu(x16), gpu(y16))
    @test l isa Float32
    @test l ≈ loss(x16, y16)
    g = gradient(loss, gpu(x16), gpu(y16))[1]
    @test g isa CuArray{eltype(x16)}
    @test cpu(g) ≈ gradient(loss, x16, y16)[1]
  end
end

@testset "GPU: $loss with class labels" for loss in (crossentropy, logitcrossentropy)
  x = softmax(randn(Float32, 5, 6))
  y = rand(1:5, 6)
  @test loss(gpu(x), gpu(y)) ≈ loss(x, y)
  @test cpu(loss(gpu(x), gpu(y); agg = identity)) ≈ loss(x, y; agg = identity)
  @test loss(gpu(x), gpu(onehotbatch(y, 1:5))) ≈ loss(x, y)
  @test_throws ArgumentError loss(gpu(x), gpu([0; y[2:end]]))
  test_gradients(loss, x, y, test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=nothing)

  @testset "$lowprec" for lowprec in (f16, bf16)
    x16 = lowprec(x)
    g = gradient(x -> loss(x, gpu(y)), gpu(x16))[1]
    @test g isa CuArray{eltype(x16)}
    @test cpu(g) ≈ gradient(x -> loss(x, y), x16)[1]
  end

  x3 = softmax(randn(Float32, 3, 5, 4); dims = 2)
  y3 = rand(1:5, 3, 4)
  @test loss(gpu(x3), gpu(y3); dims = 2) ≈ loss(x3, y3; dims = 2)
  test_gradients((x, y) -> loss(x, y; dims = 2), x3, y3, test_gpu=true, test_cpu=false, reference=AutoZygote(), compare=nothing)
end

end #testset
