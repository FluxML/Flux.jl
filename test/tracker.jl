using Flux.Tracker, Base.Test, NNlib
using Flux.Tracker: gradcheck
using NNlib

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(f(xs...)), xs...)
gradtest(f, dims...) = gradtest(f, rand.(dims)...)

@testset "Tracker" begin

@test gradtest((x, W, b) -> σ.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> σ.(W*x .+ b), (5,3), (2,5), 2)

@test gradtest((w, x) -> w'*x, randn(10, 2), randn(10))
@test gradtest((w, x) -> w*x', randn(5,5), randn(5,5))

@test gradtest(x -> sin.(sum(x, (2, 3))), (3,4,5))

@test gradtest(x -> softmax(x).*(1:3), 3)
@test gradtest(x -> softmax(x).*(1:3), (3,5))

@test gradtest(Flux.mse, rand(5,5), rand(5, 5))
@test gradtest(Flux.crossentropy, rand(5,5), rand(5, 5))

@test gradtest(x -> x', rand(5))

@test gradtest(vcat, rand(5), rand(3))
@test gradtest(vcat, rand(2,3), rand(3,3))

@testset "mean" begin
  @test gradtest(mean, rand(2, 3))

  @test gradtest(x -> mean(x, 1), rand(2, 3))
  @test gradtest(x -> mean(x, 2), rand(2, 3))
  @test gradtest(x -> mean(x, 3), rand(2, 3, 4))

  @test gradtest(x -> mean(x, [1, 2]), rand(2, 3, 4))
end

@test gradtest(x -> std(x), rand(5,5))
@test gradtest(x -> std(x, 1), rand(5,5))

@test gradtest((x, y) -> x .* y, rand(5), rand(5))

@test gradtest(rand(5)) do x
  y = x.^2
  2y + x
end

@test gradtest(conv2d, rand(10, 10, 3, 2), randn(2, 2, 3, 2))
@test gradtest(x -> maxpool2d(x, 2), rand(10, 10, 3, 2))
@test gradtest(x -> avgpool2d(x, 2), rand(10, 10, 3, 2))

@test (param([1,2,3]) .< 2) == [true, false, false]

end #testset
