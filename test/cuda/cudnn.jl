using Flux, CuArrays, Test
using Zygote
trainmode(f, x...) = forward(f, x...)[1]

@testset "CUDNN BatchNorm" begin
    @testset "4D Input" begin
        x = Float64.(collect(reshape(1:12, 2, 2, 3, 1)))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y = trainmode(m, x)
        cy = trainmode(cm, cx)

        @test cpu(data(cy)) ≈ data(y)

        g = rand(size(y)...)
        # Flux.back!(y, g)
        # Flux.back!(cy, gpu(g))

        @test m.γ ≈ cpu(cm.γ)
        @test m.β ≈ cpu(cm.β)
        @test x ≈ cpu(x)
    end

    @testset "2D Input" begin
        x = Float64.(collect(reshape(1:12, 3, 4)))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y = trainmode(m, x)
        cy = trainmode(cm, cx)

        @test cy isa CuArray{Float32,2}

        @test cpu(data(cy)) ≈ data(y)

        g = rand(size(y)...)
        #Flux.back!(y, g)
        #Flux.back!(cy, gpu(g))

        @test m.γ ≈ cpu(cm.γ)
        @test m.β ≈ cpu(cm.β)
        @test x ≈ cpu(x)
    end
end
