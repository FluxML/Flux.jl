using Flux.Losses: crossentropy, binarycrossentropy, logitbinarycrossentropy


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


@testset "GPU grad tests" begin
  x = rand(Float32, 3,3)
  y = rand(Float32, 3,3)

  for loss in ALL_LOSSES
    gpu_gradtest(loss, x, y)
  end
end

end #testset
