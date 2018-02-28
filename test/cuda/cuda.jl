using Flux, Flux.Tracker, CuArrays, Base.Test
using Flux: gpu

info("Testing Flux/GPU")

@testset "CuArrays" begin

CuArrays.allowscalar(false)

x = param(randn(5, 5))
cx = gpu(x)
@test cx isa TrackedArray && cx.data isa CuArray

x = Flux.onehotbatch([1, 2, 3], 1:3)
cx = gpu(x)
@test cx isa Flux.OneHotMatrix && cx.data isa CuArray

m = Chain(Dense(10, 5, σ), Dense(5, 2))
cm = gpu(m)

@test all(p isa TrackedArray && p.data isa CuArray for p in params(cm))
@test cm(gpu(rand(10, 10))) isa TrackedArray{Float32,2,CuArray{Float32,2}}

end

CuArrays.cudnn_available() && include("cudnn.jl")
