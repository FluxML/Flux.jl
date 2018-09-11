using Flux, Flux.Tracker, CuArrays, Test
using Flux.Tracker: TrackedArray, data

@info "Testing Flux CUDNN"

@testset "CUDNN BatchNorm" begin
    x = TrackedArray(rand(10, 10, 3, 1))
    m = BatchNorm(3)
    cx = gpu(x)
    cm = gpu(m)

    y = m(x)
    cy = cm(cx)

    @test cy isa TrackedArray{Float32,4,CuArray{Float32,4}}

    @test cpu(data(cy)) ≈ data(y)

    g = rand(size(y))
    Flux.back!(y, g)
    Flux.back!(cy, gpu(g))

    @test m.γ.grad ≈ cpu(cm.γ.grad)
    @test m.β.grad ≈ cpu(cm.β.grad)
    @test x.grad ≈ cpu(x.grad)
end
