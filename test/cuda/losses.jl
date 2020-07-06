using Flux.Losses: crossentropy, binarycrossentropy, logitbinarycrossentropy
using Zygote: pullback


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


function gpu_gradtest(f, args...)
  args_gpu = gpu.(args)
  
  l_cpu, back_cpu = pullback((args...) -> f(args...), args...)
  g_cpu = back_cpu(1)[1]
  
  l_gpu, back_gpu = pullback((args_gpu...) -> f(args_gpu...), args_gpu...)
  g_gpu = back_gpu(1)[1]
  
  @test l_cpu ≈ l_gpu
  @test g_gpu isa CuArray
  @test g_cpu ≈ collect(g_gpu) atol=1e-6
end


@testset "GPU grad tests" begin
  x = rand(Float32, 3,3)
  y = rand(Float32, 3,3)

  for loss in ALL_LOSSES
    gpu_gradtest(loss, x, y)
  end
end

end #testset