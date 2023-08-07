using Test
using Metal
using Flux
using Random, Statistics
using Zygote
Flux.gpu_backend!("Metal") # needs a restart

# include("../test_utils.jl")
include("test_utils.jl")

@testset "Basic" begin
    include("basic.jl")
end

@testset "Huber Loss test" begin
    X = Flux.gpu(Float32[1,1])
    Y = Flux.gpu(Float32[1,1])

    grad = Flux.gradient(X, Y) do a,b
        Flux.Losses.huber_loss(a,b)
    end

    @test Flux.cpu(grad[1]) == [0, 0.5]
    @test Flux.cpu(grad[2]) == [0, -0.5]
end