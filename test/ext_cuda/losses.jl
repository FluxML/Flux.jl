using Flux.Losses: crossentropy, binarycrossentropy, logitbinarycrossentropy, binary_focal_loss, focal_loss


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
  x = 0.1f0 + 0.8f0 .* rand(Float32, 3, 4)
  y = 0.1f0 + 0.8f0 .* rand(Float32, 3, 4)
  @test loss(x, y) ≈ loss(gpu(x), gpu(y))

  test_gradients(loss, x, y, test_gpu=true, test_grad_f = false)

  # Float16 tests
  @test loss(f16(x), f16(y)) ≈ loss(gpu(f16(x)), gpu(f16(y)))
  @test loss(f16(x), f16(y)) ≈ Float16(loss(x, y))  rtol=0.1  # no GPU in fact

  g16 = gradient(loss, f16(x), f16(y))[1]
  @test g16 ≈ cpu(gradient(loss, f16(gpu(x)), f16(gpu(y)))[1])
end

end #testset
