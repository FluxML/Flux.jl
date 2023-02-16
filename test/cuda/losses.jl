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

@testset "GPU ssim tests" begin
  @testset "num_dims=$N" for N=1:3
    x = rand(Float32, 16*ones(Int, N)..., 2, 2)
    y = rand(Float32, 16*ones(Int, N)..., 2, 2)
    
    @testset "$loss" for loss in (Flux.ssim, Flux.ssim_loss, Flux.ssim_loss_fast)
      @testset "cpu-gpu" begin loss(x, y) ≈ loss(gpu(x), gpu(y)) end
      @testset "autodiff" begin gpu_autodiff_test(loss, x, y) end

      # Float16 tests
      @testset "f16 cpu-gpu" begin 
        @test isapprox(loss(f16(x), f16(y)), loss(gpu(f16(x)), gpu(f16(y))), rtol=0.1) broken=(N==3) 
      end
      @testset "f16 cpu-cpu" begin 
        isapprox(loss(f16(x), f16(y)), Float16(loss(x, y)); rtol=0.1)
      end 
      @testset "f16 grad" begin 
        g16 = gradient(loss, f16(x), f16(y))[1]
        @test isapprox(g16, cpu(gradient(loss, f16(gpu(x)), f16(gpu(y)))[1]), rtol=0.1) broken=true 
      end
    end

    # sanity checks
    x = gpu(x)
    @testset "sanity check" begin
      @test Flux.ssim(x, x) ≈ 1
      @test Flux.ssim_loss(x, x) ≈ 0
      @test Flux.ssim_loss_fast(x, x) ≈ 0
    end
  end
end

@testset "GPU: $loss" for loss in ALL_LOSSES
  x = rand(Float32, 3,4)
  y = rand(Float32, 3,4)
  @test loss(x, y) ≈ loss(gpu(x), gpu(y))

  gpu_autodiff_test(loss, x, y)

  # Float16 tests
  @test loss(f16(x), f16(y)) ≈ loss(gpu(f16(x)), gpu(f16(y)))
  @test loss(f16(x), f16(y)) ≈ Float16(loss(x, y))  rtol=0.1  # no GPU in fact

  g16 = gradient(loss, f16(x), f16(y))[1]
  @test g16 ≈ cpu(gradient(loss, f16(gpu(x)), f16(gpu(y)))[1])
end

end #testset
