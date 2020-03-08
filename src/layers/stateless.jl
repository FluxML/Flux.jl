# Cost functions
"""
    mae(ŷ, y)

Return the mean of absolute error `sum(abs.(ŷ .- y)) / length(y)`
"""
mae(ŷ, y) = sum(abs.(ŷ .- y)) * 1 // length(y)


"""
    mse(ŷ, y)

Return the mean squared error `sum((ŷ .- y).^2) / length(y)`.
"""
mse(ŷ, y) = sum((ŷ .- y).^2) * 1 // length(y)


"""
    msle(ŷ, y; ϵ=eps(eltype(ŷ)))

Returns the mean of the squared logarithmic errors `sum((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2) / length(y)`.
The `ϵ` term provides numerical stability.

This error penalizes an under-predicted estimate greater than an over-predicted estimate.
"""
msle(ŷ, y; ϵ=eps(eltype(ŷ))) = sum((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2) * 1 // length(y)



"""
    huber_loss(ŷ, y; δ=1.0)

Computes the mean of the Huber loss given the prediction `ŷ` and true values `y`. By default, δ is set to 1.0.

                    | 0.5*|ŷ - y|,   for |ŷ - y| <= δ
      Hubber loss = |
                    |  δ*(|ŷ - y| - 0.5*δ),  otherwise

[`Huber Loss`](https://en.wikipedia.org/wiki/Huber_loss).
"""
function huber_loss(ŷ, y;  δ=eltype(ŷ)(1))
   abs_error = abs.(ŷ .- y)
   temp = abs_error .<  δ
   x = eltype(ŷ)(0.5)
   hub_loss = sum(((abs_error.^2) .* temp) .* x .+ δ*(abs_error .- x*δ) .* (1 .- temp)) * 1 // length(y)
end

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
  entropy = sum(y .* log.(y)) * 1 //size(y,2)
  cross_entropy = crossentropy(ŷ, y)
  return entropy + cross_entropy
end

"""
    poisson(ŷ, y)

Poisson loss function is a measure of how the predicted distribution diverges from the expected distribution.
Returns `sum(ŷ .- y .* log.(ŷ)) / size(y, 2)`

[Poisson Loss](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson).
"""
poisson(ŷ, y) = sum(ŷ .- y .* log.(ŷ)) * 1 // size(y,2)

"""
    hinge(ŷ, y)

Measures the loss given the prediction `ŷ` and true labels `y` (containing 1 or -1).
Returns `sum((max.(0, 1 .- ŷ .* y))) / size(y, 2)`

[Hinge Loss](https://en.wikipedia.org/wiki/Hinge_loss)
See also [`squared_hinge`](@ref).
"""
hinge(ŷ, y) = sum(max.(0, 1 .-  ŷ .* y)) * 1 // size(y, 2)

"""
    squared_hinge(ŷ, y)

Computes squared hinge loss given the prediction `ŷ` and true labels `y` (conatining 1 or -1).
Returns `sum((max.(0, 1 .- ŷ .* y)).^2) / size(y, 2)`

See also [`hinge`](@ref).
"""
squared_hinge(ŷ, y) = sum((max.(0, 1 .- ŷ .* y)).^2) * 1 // size(y, 2)

"""
    dice_coeff_loss(ŷ, y; smooth=1)

Loss function used in Image Segmentation. Calculates loss based on dice coefficient. Similar to F1_score.
Returns `1 - 2*sum(|ŷ .* y| + smooth) / (sum(ŷ.^2) + sum(y.^2) + smooth)`

[V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation](https://arxiv.org/pdf/1606.04797v1.pdf)
"""
dice_coeff_loss(ŷ, y; smooth=eltype(ŷ)(1.0)) = 1 - (2*sum(y .* ŷ) + smooth) / (sum(y.^2) + sum(ŷ.^2) + smooth)

"""
    tversky_loss(ŷ, y; β=0.7)

Used with imbalanced data to give more weightage to False negatives.
Larger β weigh recall higher than precision (by placing more emphasis on false negatives)
Returns `1 - sum(|y .* ŷ| + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1)`

[Tversky loss function for image segmentation using 3D fully convolutional deep networks](https://arxiv.org/pdf/1706.05721.pdf)
"""
tversky_loss(ŷ, y; β=eltype(ŷ)(0.7)) = 1 - (sum(y .* ŷ) + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1)

"""
    flatten(x::AbstractArray)

Transforms (w,h,c,b)-shaped input into (w x h x c,b)-shaped output,
by linearizing all values for each element in the batch.
"""
function flatten(x::AbstractArray)
  return reshape(x, :, size(x)[end])
end
