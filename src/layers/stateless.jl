using NNlib: logsoftmax, logσ

# Cost functions

mse(ŷ, y) = sum((ŷ .- y).^2) * 1 // length(y)

function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  -sum(y .* log.(ŷ) .* weight) * 1 // size(y, 2)
end

function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* logsoftmax(logŷ) .* weight) * 1 // size(y, 2)
end

"""
    binarycrossentropy(ŷ, y; ϵ=eps(ŷ))

Return `-y*log(ŷ + ϵ) - (1-y)*log(1-ŷ + ϵ)`. The ϵ term provides numerical stability.

    julia> binarycrossentropy.(σ.([-1.1491, 0.8619, 0.3127]), [1, 1, 0.])
    3-element Array{Float64,1}:
    1.4244
    0.352317
    0.86167
"""
binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

"""
    logitbinarycrossentropy(logŷ, y)

`logitbinarycrossentropy(logŷ, y)` is mathematically equivalent to `binarycrossentropy(σ(logŷ), y)`
but it is more numerically stable.

    julia> logitbinarycrossentropy.([-1.1491, 0.8619, 0.3127], [1, 1, 0.])
    3-element Array{Float64,1}:
     1.4244
     0.352317
     0.86167
"""
logitbinarycrossentropy(logŷ, y) = (1 - y)*logŷ - logσ(logŷ)

"""
    normalise(x::AbstractArray; dims=1)

    Normalises x to mean 0 and standard deviation 1, across the dimensions given by dims. Defaults to normalising over columns.
"""
function normalise(x::AbstractArray; dims=1)
  μ′ = mean(x, dims = dims)
  σ′ = std(x, dims = dims, mean = μ′, corrected=false)
  return (x .- μ′) ./ σ′
end

"""
    Kullback Leibler Divergence(KL Divergence)
KLDivergence is a measure of how much one probability distribution is different from the other.
It is always non-negative and zero only when both the distributions are equal everywhere.
https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
"""
function kldivergence(ŷ, y)
  entropy = sum(y .* log.(y)) *1 //size(y,2)
  cross_entropy = crossentropy(ŷ, y)
  return entropy + cross_entropy
end

"""
    Poisson Loss function
Poisson loss function is a measure of how the predicted distribution diverges from the expected distribution.
https://isaacchanghau.github.io/post/loss_functions/
"""
poisson(ŷ, y) = sum(ŷ .- y .* log.(ŷ)) *1 // size(y,2)

"""
    Hinge Loss function
Measures the loss given the prediction ŷ and true labels y(containing 1 or -1). This is usually used for measuring whether two inputs are similar or dissimilar
"""
hinge(ŷ, y) = sum(max.(0, 1 .-  ŷ .* y)) *1 // size(y,2)
