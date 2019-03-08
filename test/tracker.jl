using Flux, Test

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

gradcheck(f, xs...) =
  all(isapprox.(ngradient(f, xs...),
                gradient(f, xs...), rtol = 1e-5, atol = 1e-5))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

@testset "Tracker" begin

@test gradtest(Flux.mse, rand(5,5), rand(5, 5))
@test gradtest(Flux.crossentropy, rand(5,5), rand(5, 5))

# @test gradtest(x -> Flux.normalise(x), rand(4,3))
# @test gradtest(x -> Flux.normalise(x, dims = 2), rand(3,4))

end
