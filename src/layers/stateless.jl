# Cost functions
"""
    mae(ŷ, y)

Return the mean of absolute error; calculated as
`sum(abs.(ŷ .- y)) / length(y)`.
"""
mae(ŷ, y) = sum(abs.(ŷ .- y)) * 1 // length(y)


"""
    mse(ŷ, y)

Return the mean squared error between ŷ and y; calculated as
`sum((ŷ .- y).^2) / length(y)`.

# Examples
```jldoctest
julia> Flux.mse([0, 2], [1, 1])
1//1
```
"""
mse(ŷ, y) = sum((ŷ .- y).^2) * 1 // length(y)


"""
    msle(ŷ, y; ϵ=eps(eltype(ŷ)))

Return the mean of the squared logarithmic errors; calculated as
`sum((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2) / length(y)`.
The `ϵ` term provides numerical stability.

Penalizes an under-predicted estimate greater than an over-predicted estimate.
"""
msle(ŷ, y; ϵ=eps(eltype(ŷ))) = sum((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2) * 1 // length(y)



"""
    huber_loss(ŷ, y; δ=1.0)

Return the mean of the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
given the prediction `ŷ` and true values `y`.

                 | 0.5 * |ŷ - y|,            for |ŷ - y| <= δ
    Huber loss = |
                 |  δ * (|ŷ - y| - 0.5 * δ), otherwise
"""
function huber_loss(ŷ, y;  δ=eltype(ŷ)(1))
   abs_error = abs.(ŷ .- y)
   temp = abs_error .<  δ
   x = eltype(ŷ)(0.5)
   hub_loss = sum(((abs_error.^2) .* temp) .* x .+ δ*(abs_error .- x*δ) .* (1 .- temp)) * 1 // length(y)
end

function _crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::Nothing)
  return -sum(xlogy.(y, ŷ)) * 1 // size(y, 2)
end

function _crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::Number)
  return -sum(xlogy.(y, ŷ)) .* weight * 1 // size(y, 2)
end

function _crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::AbstractVector)
  return -sum(xlogy.(y, ŷ) .* weight) * 1 // size(y, 2)
end

"""
    crossentropy(ŷ, y; weight = nothing)

Return the cross entropy between the given probability distributions;
calculated as `-sum(y .* log.(ŷ) .* weight) / size(y, 2)`.

`weight` can be `Nothing`, a `Number` or an `AbstractVector`.
`weight=nothing` acts like `weight=1` but is faster.

See also: [`Flux.logitcrossentropy`](@ref), [`Flux.binarycrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)

# Examples
```jldoctest
julia> Flux.crossentropy(softmax([-1.1491, 0.8619, 0.3127]), [1, 1, 0])
3.085467254747739
```
"""
crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight=nothing) = _crossentropy(ŷ, y, weight)

"""
    logitcrossentropy(ŷ, y; weight = 1)

Return the crossentropy computed after a [`Flux.logsoftmax`](@ref) operation;
calculated as `-sum(y .* logsoftmax(ŷ) .* weight) / size(y, 2)`.

`logitcrossentropy(ŷ, y)` is mathematically equivalent to
[`Flux.crossentropy(softmax(ŷ), y)`](@ref) but it is more numerically stable.

See also: [`Flux.crossentropy`](@ref), [`Flux.binarycrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)

# Examples
```jldoctest
julia> Flux.logitcrossentropy([-1.1491, 0.8619, 0.3127], [1, 1, 0])
3.085467254747738
```
"""
function logitcrossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* logsoftmax(ŷ) .* weight) * 1 // size(y, 2)
end

"""
    binarycrossentropy(ŷ, y; ϵ=eps(ŷ))

Return ``-y*\\log(ŷ + ϵ) - (1-y)*\\log(1-ŷ + ϵ)``. The `ϵ` term provides numerical stability.

Typically, the prediction `ŷ` is given by the output of a [`sigmoid`](@ref) activation.

See also: [`Flux.crossentropy`](@ref), [`Flux.logitcrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)

# Examples
```jldoctest
julia> Flux.binarycrossentropy.(σ.([-1.1491, 0.8619, 0.3127]), [1, 1, 0])
3-element Array{Float64,1}:
 1.424397097347566
 0.35231664672364077
 0.8616703662235441
```
"""
binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -xlogy(y, ŷ + ϵ) - xlogy(1 - y, 1 - ŷ + ϵ)

# Re-definition to fix interaction with CuArrays.
CuArrays.@cufunc binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

"""
    logitbinarycrossentropy(ŷ, y)

`logitbinarycrossentropy(ŷ, y)` is mathematically equivalent to
[`Flux.binarycrossentropy(σ(ŷ), y)`](@ref) but it is more numerically stable.

See also: [`Flux.crossentropy`](@ref), [`Flux.logitcrossentropy`](@ref), [`Flux.binarycrossentropy`](@ref)

# Examples
```jldoctest
julia> Flux.logitbinarycrossentropy.([-1.1491, 0.8619, 0.3127], [1, 1, 0])
3-element Array{Float64,1}:
 1.4243970973475661
 0.35231664672364094
 0.8616703662235443
```
"""
logitbinarycrossentropy(ŷ, y) = (1 - y)*ŷ - logσ(ŷ)

# Re-definition to fix interaction with CuArrays.
CuArrays.@cufunc logitbinarycrossentropy(ŷ, y) = (1 - y)*ŷ - logσ(ŷ)

"""
    normalise(x; dims=1)

Normalise `x` to mean 0 and standard deviation 1 across the dimensions given by `dims`.
Defaults to normalising over columns.

```jldoctest
julia> a = reshape(collect(1:9), 3, 3)
3×3 Array{Int64,2}:
 1  4  7
 2  5  8
 3  6  9

julia> Flux.normalise(a)
3×3 Array{Float64,2}:
 -1.22474  -1.22474  -1.22474
  0.0       0.0       0.0
  1.22474   1.22474   1.22474

julia> Flux.normalise(a, dims=2)
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

Return the
[Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
between the given probability distributions.

KL divergence is a measure of how much one probability distribution is different
from the other.
It is always non-negative and zero only when both the distributions are equal
everywhere.
"""
function kldivergence(ŷ, y)
  entropy = sum(xlogx.(y)) * 1 //size(y,2)
  cross_entropy = crossentropy(ŷ, y)
  return entropy + cross_entropy
end

"""
    poisson(ŷ, y)

Return how much the predicted distribution `ŷ` diverges from the expected Poisson
distribution `y`; calculated as `sum(ŷ .- y .* log.(ŷ)) / size(y, 2)`.

[More information.](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson).
"""
poisson(ŷ, y) = sum(ŷ .- xlogy.(y, ŷ)) * 1 // size(y,2)

"""
    hinge(ŷ, y)

Return the [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) given the
prediction `ŷ` and true labels `y` (containing 1 or -1); calculated as
`sum(max.(0, 1 .- ŷ .* y)) / size(y, 2)`.

See also: [`squared_hinge`](@ref)
"""
hinge(ŷ, y) = sum(max.(0, 1 .-  ŷ .* y)) * 1 // size(y, 2)

"""
    squared_hinge(ŷ, y)

Return the squared hinge loss given the prediction `ŷ` and true labels `y`
(containing 1 or -1); calculated as `sum((max.(0, 1 .- ŷ .* y)).^2) / size(y, 2)`.

See also: [`hinge`](@ref)
"""
squared_hinge(ŷ, y) = sum((max.(0, 1 .- ŷ .* y)).^2) * 1 // size(y, 2)

"""
    dice_coeff_loss(ŷ, y; smooth=1)

Return a loss based on the dice coefficient.
Used in the [V-Net](https://arxiv.org/pdf/1606.04797v1.pdf) image segmentation
architecture.
Similar to the F1_score. Calculated as:
    1 - 2*sum(|ŷ .* y| + smooth) / (sum(ŷ.^2) + sum(y.^2) + smooth)`
"""
dice_coeff_loss(ŷ, y; smooth=eltype(ŷ)(1.0)) = 1 - (2*sum(y .* ŷ) + smooth) / (sum(y.^2) + sum(ŷ.^2) + smooth)

"""
    tversky_loss(ŷ, y; β=0.7)

Return the [Tversky loss](https://arxiv.org/pdf/1706.05721.pdf).
Used with imbalanced data to give more weight to false negatives.
Larger β weigh recall higher than precision (by placing more emphasis on false negatives)
Calculated as:
    1 - sum(|y .* ŷ| + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1)
"""
tversky_loss(ŷ, y; β=eltype(ŷ)(0.7)) = 1 - (sum(y .* ŷ) + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1)

"""
    flatten(x::AbstractArray)

Transform (w, h, c, b)-shaped input into (w × h × c, b)-shaped output
by linearizing all values for each element in the batch.
"""
function flatten(x::AbstractArray)
  return reshape(x, :, size(x)[end])
end

"""
    xlogx(x)
Return `x * log(x)` for `x ≥ 0`, handling `x = 0` by taking the downward limit.
"""
function xlogx(x)
  result = x * log(x)
  ifelse(iszero(x), zero(result), result)
end
CuArrays.@cufunc function xlogx(x)
  result = x * log(x)
  ifelse(iszero(x), zero(result), result)
end

"""
    xlogy(x, y)
Return `x * log(y)` for `y > 0` with correct limit at `x = 0`.
"""
function xlogy(x, y)
  result = x * log(y)
  ifelse(iszero(x), zero(result), result)
end
CuArrays.@cufunc function xlogy(x, y)
  result = x * log(y)
  ifelse(iszero(x), zero(result), result)
end

@adjoint function broadcasted(::typeof(xlogy), x::Zygote.Numeric, y::Zygote.Numeric)
  res = xlogy.(x, y)
  res, Δ -> (nothing, Zygote.unbroadcast(x, xlogy.(Δ, y)), Zygote.unbroadcast(y, Δ .* x ./ y))
end
