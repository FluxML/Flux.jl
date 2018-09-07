using NNlib: logsoftmax, logσ

# Cost functions

mse(ŷ, y) = sum((ŷ .- y).^2)/length(y)

function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  -sum(y .* log.(ŷ) .* weight) / size(y, 2)
end

@deprecate logloss(x, y) crossentropy(x, y)

function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* logsoftmax(logŷ) .* weight) / size(y, 2)
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
    normalise(x::AbstractVecOrMat)

Normalise each column of `x` to mean 0 and standard deviation 1.
"""
function normalise(x::AbstractVecOrMat)
  μ′ = mean(x, dims = 1)
  σ′ = std(x, dims = 1, mean = μ′)
  return (x .- μ′) ./ σ′
end
