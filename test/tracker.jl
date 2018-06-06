using Flux.Tracker, Base.Test, NNlib
using Flux.Tracker: TrackedReal, gradcheck, grad
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

function promotiontest(f, A, B, C)
  r0 = f(A, B, C)
  r1 = f(param(A), B, C)
  r2 = f(A, param(B), C)
  if all(ndims.((A,B,C)) .≤ 2) && f ∈ [hcat, vcat]
    r3 = f(A, B, param(C))
  else
    @test_throws MethodError f(A, B, param(C)) # until julia#20815 is resolved
    r3 = r2
  end
  r4 = f(param(A), param(B), param(C))

  @test !isa(r0, TrackedArray)
  @test all(isa.([r1,r2,r3,r4], TrackedArray))
  @test r1 == r2 == r3 == r4
  @test r0 == Flux.data(r4)
end

@testset "concat" begin
  cat1(x...) = cat(1, x...)
  cat2(x...) = cat(2, x...)

  @testset for vcatf in [vcat, cat1]
    @test gradtest(vcatf, rand(5), rand(3))
    @test gradtest(vcatf, rand(5), rand(3), rand(8))
    @test gradtest(vcatf, rand(5)', rand(5)')
    @test gradtest(vcatf, rand(5,2), rand(3,2), rand(8,2))
    @test gradtest(vcatf, rand(5,2,3), rand(3,2,3), rand(8,2,3))
    @test gradtest(vcatf, rand(5), rand(3,1))
    @test gradtest(vcatf, rand(5)', rand(2,5))
  end

  @testset for hcatf in [hcat, cat2]
    @test gradtest(hcatf, rand(5), rand(5))
    @test gradtest(hcatf, rand(5)', rand(5)')
    @test gradtest(hcatf, rand(2,5), rand(2,3), rand(2,8))
    @test gradtest(hcatf, rand(2,5,3), rand(2,3,3), rand(2,8,3))
    @test gradtest(hcatf, rand(5), rand(5), rand(5,2))
    @test gradtest(hcatf, rand(5)', rand(1,3))
    @test gradtest(hcatf, rand(5), rand(5,2))
end

  @testset for catf in [vcat, cat1, hcat, cat2, (x...) -> cat(3, x...), (x...) -> cat((1,2), x...)]
    @test gradtest(catf, rand(5))
    @test gradtest(catf, rand(5)')
    @test gradtest(catf, rand(2,5))
    @test gradtest(catf, rand(2,5,3))
  end

  @test gradtest((x...) -> cat(3, x...), rand(2,5,2), rand(2,5,3), rand(2,5,4))

  @testset "cat($dim, ...)" for dim in 3:5
    catdim = (x...) -> cat(dim, x...)
    @test gradtest(catdim, rand(5), rand(5), rand(5))
    @test gradtest(catdim, rand(2,5), rand(2,5), rand(2,5))
    @test gradtest(catdim, rand(2,5,3), rand(2,5,3), rand(2,5,3))
  end

  @test !isa(vcat(rand(2)), TrackedArray)
  @test !isa(hcat(rand(2)), TrackedArray)
  @test !isa(cat(1,rand(2)), TrackedArray)

  @test gradtest((a,b)->cat((2,3,5), a, b), rand(2,3), rand(2,4,2,1))

  @testset "promotiontest" begin
    @testset for fcat in [hcat, vcat, (x...) -> cat(3, x...), (x...) -> cat((1,2), x...)]
      promotiontest(fcat, rand(2), rand(2), rand(2))
      promotiontest(fcat, rand(2)', rand(2)', rand(2)')
      promotiontest(fcat, rand(2,2), rand(2,2), rand(2,2))
      promotiontest(fcat, rand(2,2,2), rand(2,2,2), rand(2,2,2))
    end

    promotiontest(vcat, rand(1,2), rand(2)', rand(2,2))
    promotiontest(hcat, rand(2,1), rand(2), rand(2,2))
    promotiontest(vcat, rand(3,4,5), rand(1,4,5), rand(2,4,5))
    promotiontest(hcat, rand(4,3,5), rand(4,1,5), rand(4,2,5))
    promotiontest((x...) -> cat(3, x...), rand(4,5,3), rand(4,5,1), rand(4,5,2))
  end
end

@test gradtest(x -> permutedims(x, [3,1,2]), rand(4,5,6))

@test gradtest(x -> repmat(x, 5,5), rand(4,5))
@test gradtest(x -> repmat(x, 5), rand(4,5))

@test gradtest(kron, rand(5), rand(3))
@test gradtest(kron, rand(5), rand(3), rand(8))
@test gradtest(kron, rand(5,1), rand(3,1))
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

@testset "maximum" begin
  @test gradtest(maximum, rand(2, 3))

  @test gradtest(x -> maximum(x, 1), rand(2, 3))
  @test gradtest(x -> maximum(x, 2), rand(2, 3))
  @test gradtest(x -> maximum(x, 3), rand(2, 3, 4))

  @test gradtest(x -> maximum(x, [1, 2]), rand(2, 3, 4))
end

@testset "minimum" begin
  @test gradtest(minimum, rand(2, 3))

  @test gradtest(x -> minimum(x, 1), rand(2, 3))
  @test gradtest(x -> minimum(x, 2), rand(2, 3))
  @test gradtest(x -> minimum(x, 3), rand(2, 3, 4))

  @test gradtest(x -> minimum(x, [1, 2]), rand(2, 3, 4))
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

@testset "reshape" begin
  x = reshape(param(rand(2,2,2)), 4, 2)
  @test x isa TrackedArray
  @test size(x) == (4,2)
  x = reshape(param([1]), (1,:))
  @test x isa TrackedArray
  @test size(x) == (1,1)
  x = reshape(param(rand(2)), (2,:))
  @test x isa TrackedArray
  @test size(x) == (2,1)
  x = reshape(param(rand(2,2)), (1,:,2))
  @test x isa TrackedArray
  @test size(x) == (1,2,2)
end

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

b = param(rand())
Tracker.back!(b)
@test Tracker.grad(b) == 1

@testset "collect" begin
  x, y = param(2), param(3)
  xy = Tracker.collect([x, y])
  @test xy isa TrackedArray{Float64}
  z = xy[1]*xy[2]
  back!(z)
  @test grad.((x,y)) == (3, 2)
end

end #testset
