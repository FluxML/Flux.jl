using Flux, Flux.Tracker, CuArrays, Test
using Flux.Tracker: TrackedArray, data

@testset "CUDNN BatchNorm" begin
    @testset "4D Input" begin
        x = TrackedArray(Float64.(collect(reshape(1:12, 2, 2, 3, 1))))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y = m(x)
        cy = cm(cx)

        @test cy isa TrackedArray{Float32,4,CuArray{Float32,4}}

        @test cpu(data(cy)) ≈ data(y)

        g = rand(size(y)...)
        Flux.back!(y, g)
        Flux.back!(cy, gpu(g))

        @test m.γ.grad ≈ cpu(cm.γ.grad)
        @test m.β.grad ≈ cpu(cm.β.grad)
        @test x.grad ≈ cpu(x.grad)
    end

    @testset "2D Input" begin
        x = TrackedArray(Float64.(collect(reshape(1:12, 3, 4))))
        m = BatchNorm(3)
        cx = gpu(x)
        cm = gpu(m)

        y = m(x)
        cy = cm(cx)

        @test cy isa TrackedArray{Float32,2,CuArray{Float32,2}}

        @test cpu(data(cy)) ≈ data(y)

        g = rand(size(y)...)
        Flux.back!(y, g)
        Flux.back!(cy, gpu(g))

        @test m.γ.grad ≈ cpu(cm.γ.grad)
        @test m.β.grad ≈ cpu(cm.β.grad)
        @test x.grad ≈ cpu(x.grad)
    end
end
