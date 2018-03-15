using Flux.Tracker, Base.Test, NNlib
using Flux.Tracker: TrackedReal, gradcheck
using NNlib: conv

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(dims)...)

@testset "Tracker" begin

@test gradtest((x, W, b) -> σ.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> σ.(W*x .+ b), (5,3), (2,5), 2)
@test gradtest((x, W, b) -> logσ.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> logσ.(W*x .+ b), (5,3), (2,5), 2)

@test gradtest((w, x) -> w'*x, randn(10, 2), randn(10))
@test gradtest((w, x) -> w*x', randn(5,5), randn(5,5))

@test gradtest(x -> sum(x, (2, 3)), (3,4,5))
@test gradtest(x -> prod(x, (2, 3)), (3,4,5))
@test gradtest(x -> prod(x), (3,4,5))

@test gradtest(x -> softmax(x).*(1:3), 3)
@test gradtest(x -> softmax(x).*(1:3), (3,5))
@test gradtest(x -> logsoftmax(x).*(1:3), 3)
@test gradtest(x -> logsoftmax(x).*(1:3), (3,5))

@test gradtest(Flux.mse, rand(5,5), rand(5, 5))
@test gradtest(Flux.crossentropy, rand(5,5), rand(5, 5))

@test gradtest(x -> x', rand(5))

@test gradtest(vcat, rand(5), rand(3))
@test gradtest(vcat, rand(5), rand(3), rand(8))
@test gradtest(vcat, rand(5,2), rand(3,2), rand(8,2))
@test gradtest(x -> permutedims(x, [3,1,2]), rand(4,5,6))

@test gradtest(kron,rand(5), rand(3))
@test gradtest(kron, rand(5), rand(3), rand(8))
@test gradtest(kron,rand(5,1), rand(3,1))
@test gradtest(kron, rand(5,1), rand(3,1), rand(8,1))
@test gradtest(kron, rand(5,2), rand(3,2), rand(8,2))

@test gradtest(diagm, rand(3))

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
@test gradtest(dot, rand(5), rand(5))

@test gradtest(vecnorm, rand(5))

@test gradtest(rand(5)) do x
  y = x.^2
  2y + x
end

@test gradtest(conv, rand(10, 3, 2), randn(2, 3, 2))
@test gradtest(conv, rand(10, 10, 3, 2), randn(2, 2, 3, 2))
@test gradtest(conv, rand(10, 10, 10, 3, 2), randn(2, 2, 2, 3, 2))

@test gradtest(x -> maxpool(x, (2,2)), rand(10, 10, 3, 2))
@test gradtest(x -> maxpool(x, (2,2,2)), rand(10, 10, 10, 3, 2))

@test gradtest(x -> meanpool(x, (2,2)), rand(10, 10, 3, 2))
@test gradtest(x -> meanpool(x, (2,2,2)), rand(5, 5, 5, 3, 2))

@test (param([1,2,3]) .< 2) == [true, false, false]

@test param(2)^2 == 4.0

@testset "Intermediates" begin
  x = param([1])
  l = sum((x .+ x).^2)
  Flux.back!(l)
  @test x.grad == [8]
  x.grad .= 0
  Flux.back!(l)
  @test x.grad == [8]
end

@testset "Fallbacks" begin
  xs = param([1 2; 3 4])
  @test similar(xs) isa Matrix{Float64}
  # Remove this test if we do LowerTriangular properly
  L = LowerTriangular(xs)
  @test L*L' isa Matrix{TrackedReal{Float64}}
end

@test @sprintf("%.2f", sum(param([1,2,3]))) == "6.00"

@inferred NNlib.conv(param(rand(10,10,3,2)),randn(2,2,3,4))

end #testset
