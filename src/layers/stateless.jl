# Cost functions
"""
    mse(ŷ, y)

Return the mean squared error `sum((ŷ .- y).^2) / length(y)`. 
"""
mse(ŷ, y) = sum((ŷ .- y).^2) * 1 // length(y)


function _crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::Nothing)
  return -sum(y .* log.(ŷ)) * 1 // size(y, 2)
end

function _crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::Number)
  return -sum(y .* log.(ŷ)) .* weight * 1 // size(y, 2)
end

function _crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::AbstractVector)
  return -sum(y .* log.(ŷ) .* weight) * 1 // size(y, 2)
end

"""
    crossentropy(ŷ, y; weight=1)

Return the crossentropy computed as `-sum(y .* log.(ŷ) .* weight) / size(y, 2)`. 

See also [`logitcrossentropy`](@ref), [`binarycrossentropy`](@ref).
"""
crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight=nothing) = _crossentropy(ŷ, y, weight)

"""
    logitcrossentropy(ŷ, y; weight=1)

Return the crossentropy computed after a [softmax](@ref) operation: 

  -sum(y .* logsoftmax(ŷ) .* weight) / size(y, 2)

See also [`crossentropy`](@ref), [`binarycrossentropy`](@ref).
"""
function logitcrossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* logsoftmax(ŷ) .* weight) * 1 // size(y, 2)
end

"""
    binarycrossentropy(ŷ, y; ϵ=eps(ŷ))

Return `-y*log(ŷ + ϵ) - (1-y)*log(1-ŷ + ϵ)`. The ϵ term provides numerical stability.

Typically, the prediction `ŷ` is given by the output of a [`sigmoid`](@ref) activation.
"""
binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

# Re-definition to fix interaction with CuArrays.
CuArrays.@cufunc binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

"""
    logitbinarycrossentropy(ŷ, y)

`logitbinarycrossentropy(ŷ, y)` is mathematically equivalent to `binarycrossentropy(σ(ŷ), y)`
but it is more numerically stable.

See also [`binarycrossentropy`](@ref), [`sigmoid`](@ref), [`logsigmoid`](@ref).  
"""
logitbinarycrossentropy(ŷ, y) = (1 - y)*ŷ - logσ(ŷ)

# Re-definition to fix interaction with CuArrays.
CuArrays.@cufunc logitbinarycrossentropy(ŷ, y) = (1 - y)*ŷ - logσ(ŷ)

"""
    normalise(x; dims=1)

Normalises `x` to mean 0 and standard deviation 1, across the dimensions given by `dims`. Defaults to normalising over columns.

```julia-repl
julia> a = reshape(collect(1:9), 3, 3)
3×3 Array{Int64,2}:
  1  4  7
  2  5  8
  3  6  9

julia> normalise(a)
3×3 Array{Float64,2}:
  -1.22474  -1.22474  -1.22474
  0.0       0.0       0.0
  1.22474   1.22474   1.22474

julia> normalise(a, dims=2)
3×3 Array{Float64,2}:
  -1.22474  0.0  1.22474
  -1.22474  0.0  1.22474
  -1.22474  0.0  1.22474
```
"""
function normalise(x::AbstractArray; dims=1)
  μ′ = mean(x, dims = dims)
  σ′ = std(x, dims = dims, mean = μ′, corrected=false)
  return (x .- μ′) ./ σ′
end

"""
    kldivergence(ŷ, y)

KLDivergence is a measure of how much one probability distribution is different from the other.
It is always non-negative and zero only when both the distributions are equal everywhere.
[KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
"""
function kldivergence(ŷ, y)
  entropy = sum(y .* log.(y)) *1 //size(y,2)
  cross_entropy = crossentropy(ŷ, y)
  return entropy + cross_entropy
end

"""
    poisson(ŷ, y)

Poisson loss function is a measure of how the predicted distribution diverges from the expected distribution.
[Poisson Loss](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson).
"""
poisson(ŷ, y) = sum(ŷ .- y .* log.(ŷ)) *1 // size(y,2)

"""
    hinge(ŷ, y)

Measures the loss given the prediction `ŷ` and true labels `y` (containing 1 or -1). 
[Hinge Loss](https://en.wikipedia.org/wiki/Hinge_loss).
"""
hinge(ŷ, y) = sum(max.(0, 1 .-  ŷ .* y)) *1 // size(y,2)
