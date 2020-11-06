# v0.11 deprecations
@deprecate poisson poisson_loss false
@deprecate hinge hinge_loss false
@deprecate squared_hinge squared_hinge_loss false
@deprecate normalise(x) normalise(x, dims=1) false

@deprecate binarycrossentropy(ŷ, y) Losses.binarycrossentropy(ŷ, y, agg=identity) false
@deprecate logitbinarycrossentropy(ŷ, y) Losses.logitbinarycrossentropy(ŷ, y, agg=identity) false

function Broadcast.broadcasted(::typeof(binarycrossentropy), ŷ, y)
    @warn "binarycrossentropy.(ŷ, y) is deprecated, use Losses.binarycrossentropy(ŷ, y, agg=identity) instead"
    Losses.binarycrossentropy(ŷ, y, agg=identity)
end

function Broadcast.broadcasted(::typeof(logitbinarycrossentropy), ŷ, y)
    @warn "logitbinarycrossentropy.(ŷ, y) is deprecated, use Losses.logitbinarycrossentropy(ŷ, y, agg=identity) instead"
    Losses.logitbinarycrossentropy(ŷ, y, agg=identity)
end


# v0.12 deprecations
@deprecate Dropout(p, dims) Dropout(p; dims=dims)
@deprecate InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum) InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum, nothing)
@deprecate BatchNorm(λ, β, γ, μ, σ², ϵ, momentum) BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, nothing)
@deprecate GroupNorm(G, λ, β, γ, μ, σ², ϵ, momentum) GroupNorm(G, λ, β, γ, μ, σ², ϵ, momentum, nothing)
