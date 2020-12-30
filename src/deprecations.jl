# v0.12 deprecations
@deprecate Dropout(p, dims) Dropout(p; dims=dims)
@deprecate InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum) InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum, nothing)
@deprecate BatchNorm(λ, β, γ, μ, σ², ϵ, momentum) BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, nothing)
@deprecate GroupNorm(G, λ, β, γ, μ, σ², ϵ, momentum) GroupNorm(G, λ, β, γ, μ, σ², ϵ, momentum, nothing)
@deprecate outdims(f, inputsize) outputsize(f, inputsize)
