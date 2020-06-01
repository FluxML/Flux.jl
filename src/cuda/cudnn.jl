import ..Flux: data
import CUDA.CUDNN: batchnorm, ∇batchnorm

(BN::Flux.BatchNorm)(x::Union{CuArray{T,2},CuArray{T,4},CuArray{T,5}}, cache = nothing) where T<:Union{Float32, Float64} =
  BN.λ.(batchnorm(BN.γ, BN.β, x, BN.μ, BN.σ², BN.momentum; cache = cache, alpha = 1, beta = 0, eps = BN.ϵ, training = Flux.istraining()))

@adjoint batchnorm(g, b, x, running_mean, running_var, momentum; kw...) =
  batchnorm(g, b, x, running_mean, running_var, momentum; kw...), Δ -> (∇batchnorm(g, b, x, Δ, running_mean, running_var, momentum; kw...)..., nothing, nothing, nothing)
