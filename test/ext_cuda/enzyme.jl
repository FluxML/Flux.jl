@testset "CUDA Enzyme gradients" begin
  model = Chain(Dense(3 => 4, relu), Dense(4 => 2))
  x = randn(Float32, 3, 4)
  y = Float32[1 0 0 1;
              0 1 1 0]

  loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

  # Compare Enzyme's CUDA gradient against the CPU Zygote reference.
  gz = first(Flux.gradient(loss, AutoZygote(), model, x, y))
  model_gpu, x_gpu, y_gpu = gpu.((model, x, y))
  @test loss(model, x, y) ≈ loss(model_gpu, x_gpu, y_gpu)

  ge = first(Flux.gradient(loss, AutoEnzyme(), model_gpu, Const(x_gpu), Const(y_gpu)))
  check_equal_leaves(gz, ge |> cpu)
end
