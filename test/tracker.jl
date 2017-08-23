using Flux.Tracker: gradcheck
using Base.Test, NNlib

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(f(xs...)), xs...)
gradtest(f, dims...) = gradtest(f, rand.(dims)...)

@testset "Tracker" begin

@test gradtest((x, W, b) -> σ.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> σ.(W*x .+ b), (5,3), (2,5), 2)

@test gradtest(x -> sin.(sum(x, (2, 3))), (3,4,5))

end
