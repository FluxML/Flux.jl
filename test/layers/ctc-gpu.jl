using Test
using Flux
using Flux: ctc
using Flux.Tracker: gradient
using LinearAlgebra
using CuArrays
using Statistics

function ctc_ngradient(xs...)
  f = ctc
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    t = div(i-1, size(x, 1)) + 1
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)[t]
    x[i] = tmp + δ/2
    y2 = f(xs...)[t]
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

@testset "ctc-gpu" begin
  
  x = rand(10, 50)
  y = reduce(hcat, repeat([Array{Float64}(I, 10, 10)[min(i, 9),:] for i in 1:10], inner=5))
  
  x_cu = CuArray(x)
  y_cu = CuArray(y)
  
  g1 = Flux.Tracker.gradient(ctc, x_cu, y_cu)[1]
  g1 = Flux.Tracker.data(g1) |> collect
  
  g2 = ctc_ngradient(x, y)[1]
  
  @test all(isapprox.(g1, g2, rtol=1e-5, atol=1e-5))
  
  l1 = Flux.ctc_(x_cu, y_cu)[1]
  l2 = Flux.ctc_(x, y)[1]
  
  @test all(isapprox.(l1, l2, rtol=1e-5, atol=1e-5))
  
  x_cu = [1. 2. 3.; 2. 1. 1.; 3. 3. 2.] |> CuArray
  y_cu = [1 1 0; 0 0 1; 0 0 0] |> CuArray
  
  @test mean(ctc(x_cu, y_cu)) ≈ 3.6990738275138035
  
  g = [-0.317671 -0.427729 0.665241; 0.244728 -0.0196172 -0.829811; 0.0729422 0.447346 0.16457]
  ghat = gradient(ctc, x_cu, y_cu)[1] |> collect
  
  @test all(isapprox.(g, ghat, rtol=1e-5, atol=1e-5))
  
end
