using Flux.Losses: crossentropy, binarycrossentropy, logitbinarycrossentropy, binary_focal_loss, focal_loss,
                   margin_ranking_loss


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

@testset "GPU grad tests" begin
  x = rand(Float32, 3,3)
  y = rand(Float32, 3,3)

  for loss in ALL_LOSSES
    gpu_autodiff_test(loss, x, y)
  end
end

@testset "margin_ranking_loss" begin
     x1 = [1.2 2.3 3.4]
     x2 = [9.8 8.7 7.6]
     y = 1
   
     @test Flux.margin_ranking_loss(x1, x2, y) ≈ Flux.margin_ranking_loss(gpu(x1), gpu(x2), gpu(y)) |> cpu
     @test Flux.margin_ranking_loss(x1, x2, y, margin=1.0) ≈ 
          Flux.margin_ranking_loss(gpu(x1), gpu(x2), gpu(y), margin=gpu(1.0)) |> cpu atol=1e-6
     @test Flux.margin_ranking_loss(x1, x2, y, margin=1.0, agg=sum) ≈
          Flux.margin_ranking_loss(gpu(x1), gpu(x2), gpu(y), margin=gpu(1.0), agg=sum)
end

end #testset
