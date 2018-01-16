using Flux, Flux.Tracker, CuArrays, Base.Test

@testset "CuArrays" begin

CuArrays.allowscalar(false)

x = param(randn(5, 5))
cx = cu(x)
@test cx isa TrackedArray && cx.data isa CuArray

x = Flux.onehotbatch([1, 2, 3], 1:3)
cx = cu(x)
@test cx isa Flux.OneHotMatrix && cx.data isa CuArray

m = Chain(Dense(10, 5, σ), Dense(5, 2))
cm = cu(m)

@test all(p isa TrackedArray && p.data isa CuArray for p in params(cm))
@test cm(cu(rand(10, 10))) isa TrackedArray{Float32,2,CuArray{Float32,2}}

end
