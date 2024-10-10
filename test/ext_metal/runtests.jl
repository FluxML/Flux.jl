using Test
using Metal
using Flux
using Random, Statistics
using Zygote
Flux.gpu_backend!("Metal") # needs a restart

include("test_utils.jl")

@testset "data movement" begin
    metal_device = Flux.gpu_device()
    cdev = cpu_device()

    @test metal_device isa Flux.MetalDevice

    x = randn(Float32, 5, 5)
    cx = x |> metal_device
    @test cx isa Metal.MtlMatrix{Float32}
    x2 = cx |> cdev
    @test x2 isa Matrix{Float32}
    @test x ≈ x2
    
    metal_device = gpu_device(1)
    @test metal_device isa Flux.MetalDevice

    @test cpu(cx) isa Matrix{Float32}
    @test cpu(cx) ≈ x

    @test gpu(x) isa Metal.MtlMatrix{Float32}
    @test cpu(gpu(x)) ≈ x
end

@testset "Basic" begin
    include("basic.jl")
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
