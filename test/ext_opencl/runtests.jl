@testset "data movement" begin
    opencl_device = Flux.gpu_device()
    cdev = cpu_device()

    @test opencl_device isa Flux.OpenCLDevice

    x = randn(Float32, 5, 5)
    cx = x |> opencl_device
    @test cx isa OpenCL.CLMatrix{Float32}
    x2 = cx |> cdev
    @test x2 isa Matrix{Float32}
    @test x ≈ x2
    
    opencl_device = gpu_device(1)
    @test opencl_device isa Flux.OpenCLDevice

    @test cpu(cx) isa Matrix{Float32}
    @test cpu(cx) ≈ x

    @test gpu(x) isa OpenCL.CLMatrix{Float32}
    @test cpu(gpu(x)) ≈ x
end

@testset "Basic" begin
    include("basic.jl")
end

@testset "Recurrent" begin
    global BROKEN_TESTS = [:lstm, :gru, :gruv3]
    include("../ext_common/recurrent_gpu_ad.jl")
end

@testset "Huber Loss test" begin
    X = Flux.gpu(Float32[0,1])
    Y = Flux.gpu(Float32[1,0])

    grad = Flux.gradient(X, Y) do a,b
        Flux.Losses.huber_loss(a,b)
    end

    @test Flux.cpu(grad[1]) == [-0.5, 0.5]
    @test Flux.cpu(grad[2]) == [0.5, -0.5]
end

