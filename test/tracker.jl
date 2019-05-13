using Flux, Test
using Tracker: gradcheck

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

@testset "Tracker" begin

@test gradtest(Flux.mse, rand(5,5), rand(5, 5))
@test gradtest(Flux.crossentropy, rand(5,5), rand(5, 5))

@test gradtest(x -> Flux.normalise(x), rand(4,3))
@test gradtest(x -> Flux.normalise(x, dims = 2), rand(3,4))

end
