using Flux, Flux.Tracker, CuArrays, Base.Test
using Flux.Tracker: TrackedArray
using Flux: gpu

@testset "CUDNN BatchNorm" begin
    x = TrackedArray(rand(10, 10, 3, 1))
    m = BatchNorm(3)
    cx = gpu(x)
    cm = gpu(m)

    y = m(x)
    cy = cm(cx)

    @test cy isa TrackedArray{Float32,4,CuArray{Float32,4}}

    @test cpu(cy) ≈ y

    Flux.back!(y, ones(y))
    Flux.back!(cy, ones(cy))

    @test m.γ.grad ≈ cpu(cm.γ.grad)
    @test m.β.grad ≈ cpu(cm.β.grad)
    @test m.x.grad ≈ cpu(cm.x.grad)
end
