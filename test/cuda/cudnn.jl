using Flux, Flux.Tracker, CuArrays, Base.Test
using Flux: gpu

@testset "CUDNN BatchNorm" begin
    x = gpu(rand(10, 10, 3, 1))
    m = gpu(BatchNorm(3))
    @test m(x) isa TrackedArray{Float32,4,CuArray{Float32,4}}
end
