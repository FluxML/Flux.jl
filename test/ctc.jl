using Test
using Flux
using Flux: ctc_
using Zygote: gradient
using LinearAlgebra

# Custom function to check numerical gradient of ctc loss,
# based on `ngradient` in `Tracker.jl`
# 
# Needs to check loss as defined at a particular time step
# related to the change in x because slight deviations in
# input propagate through further time steps, intrinsically
# causing the gradients to change and thus not be comparable
# between the numeric and analytical definitions
function ctc_ngradient(xs...)
  f = ctc_
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    t = div(i-1, size(x, 1)) + 1
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)[1][t]
    x[i] = tmp + δ/2
    y2 = f(xs...)[1][t]
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

@testset "ctc" begin
  
  x = rand(10, 50)
  y = reduce(hcat, repeat([Array{Float64}(I, 10, 10)[min(i, 9),:] for i in 1:10], inner=5))
  
  g1 = gradient(ctc, x, y)[1]
  g1 = g1
  g2 = ctc_ngradient(x, y)[1]
  
  @test all(isapprox.(g1, g2, rtol=1e-5, atol=1e-5))
  
  # tests using hand-calculated values
  
  x = [1. 2. 3.; 2. 1. 1.; 3. 3. 2.]
  y = [1 1 0; 0 0 1; 0 0 0]
  
  @test ctc(x, y) ≈ 3.6990738275138035
  g = [-0.317671 -0.427729 0.665241; 0.244728 -0.0196172 -0.829811; 0.0729422 0.447346 0.16457]
  ghat = gradient(ctc, x, y)[1]
  
  @test all(isapprox.(g, ghat, rtol=1e-5, atol=1e-5))
  
end
