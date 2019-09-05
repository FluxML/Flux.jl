using Test
using Flux
using Flux: ctc
using Flux.Tracker: gradient
using CUDAapi: has_cuda
using LinearAlgebra

# Custom function to check numerical gradient of ctc loss,
# based on `ngradient` in `Flux.Tracker`
# 
# Needs to check loss as defined at a particular time step
# related to the change in x because slight deviations in
# input propagate through further time steps, intrinsically
# causing the gradients to change and thus not be comparable
# between the numeric and analytical definitions
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

@testset "ctc" begin
  
  x = rand(10, 50)
  y = reduce(hcat, repeat([Array{Float64}(I, 10, 10)[min(i, 9),:] for i in 1:10], inner=5))
  
  g1 = Flux.Tracker.gradient(ctc, x, y)[1]
  g1 = Flux.Tracker.data(g1)
  g2 = ctc_ngradient(x, y)[1]
  
  @test all(isapprox.(g1, g2, rtol=1e-5, atol=1e-5))
  
  if has_cuda()
    include("ctc-gpu.jl")
  end
  
end
