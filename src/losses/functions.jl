# In this file, doctests which differ in the printed Float32 values won't fail
```@meta
DocTestFilters = r"[0-9\.]+f0"
```

"""
    mae(ŷ, y; agg = mean)

Return the loss corresponding to mean absolute error:

    agg(abs.(ŷ .- y))

# Example
```jldoctest
julia> y_model = [1.1, 1.9, 3.1];

julia> Flux.mae(y_model, 1:3)
0.10000000000000009
```
"""
mae(ŷ, y; agg = mean) = agg(abs.(ŷ .- y))

"""
    mse(ŷ, y; agg = mean)

Return the loss corresponding to mean square error:

    agg((ŷ .- y) .^ 2)

See also: [`mae`](@ref), [`msle`](@ref), [`crossentropy`](@ref).

# Example
```jldoctest
julia> y_model = [1.1, 1.9, 3.1];

julia> y_true = 1:3;

julia> Flux.mse(y_model, y_true)
0.010000000000000018
```
"""
mse(ŷ, y; agg = mean) = agg((ŷ .- y) .^ 2)

"""
    msle(ŷ, y; agg = mean, ϵ = eps(ŷ))

The loss corresponding to mean squared logarithmic errors, calculated as

    agg((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)) .^ 2)

The `ϵ` term provides numerical stability.
Penalizes an under-estimation more than an over-estimatation.

# Example
```jldoctest
julia> Flux.msle(Float32[1.1, 2.2, 3.3], 1:3)
0.009084041f0

julia> Flux.msle(Float32[0.9, 1.8, 2.7], 1:3)
0.011100831f0
```
"""
msle(ŷ, y; agg = mean, ϵ = epseltype(ŷ)) =
  agg((log.((ŷ .+ ϵ) ./ (y .+ ϵ))) .^2 )

"""
    huber_loss(ŷ, y; δ = 1, agg = mean)

Return the mean of the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
given the prediction `ŷ` and true values `y`.

                 | 0.5 * |ŷ - y|^2,            for |ŷ - y| <= δ
    Huber loss = |
                 |  δ * (|ŷ - y| - 0.5 * δ), otherwise
"""
function huber_loss(ŷ, y; agg = mean, δ = ofeltype(ŷ, 1))
   abs_error = abs.(ŷ .- y)
   #TODO: remove dropgrad when Zygote can handle this function with CuArrays
   temp = Zygote.dropgrad(abs_error .<  δ)
   x = ofeltype(ŷ, 0.5)
   agg(((abs_error .^ 2) .* temp) .* x .+ δ * (abs_error .- x * δ) .* (1 .- temp))
end

"""
    label_smoothing(y::Union{Number, AbstractArray}, α; dims::Int=1)

Returns smoothed labels, meaning the confidence on label values are relaxed.

When `y` is given as one-hot vector or batch of one-hot, its calculated as

    y .* (1 - α) .+ α / size(y, dims)

when `y` is given as a number or batch of numbers for binary classification,
its calculated as

    y .* (1 - α) .+ α / 2

in which case the labels are squeezed towards `0.5`.

α is a number in interval (0, 1) called the smoothing factor. Higher the
value of α larger the smoothing of `y`.

`dims` denotes the one-hot dimension, unless `dims=0` which denotes the application
of label smoothing to binary distributions encoded in a single number.

# Example
```jldoctest
julia> y = Flux.onehotbatch([1, 1, 1, 0, 1, 0], 0:1)
2×6 Flux.OneHotArray{UInt32,2,1,2,Array{UInt32,1}}:
 0  0  0  1  0  1
 1  1  1  0  1  0

julia> y_smoothed = Flux.label_smoothing(y, 0.2f0)
2×6 Array{Float32,2}:
 0.1  0.1  0.1  0.9  0.1  0.9
 0.9  0.9  0.9  0.1  0.9  0.1

julia> y_sim = softmax(y .* log(2f0))
2×6 Array{Float32,2}:
 0.333333  0.333333  0.333333  0.666667  0.333333  0.666667
 0.666667  0.666667  0.666667  0.333333  0.666667  0.333333

julia> y_dis = vcat(y_sim[2,:]', y_sim[1,:]')
2×6 Array{Float32,2}:
 0.666667  0.666667  0.666667  0.333333  0.666667  0.333333
 0.333333  0.333333  0.333333  0.666667  0.333333  0.666667

julia> Flux.crossentropy(y_sim, y) < Flux.crossentropy(y_sim, y_smoothed)
true

julia> Flux.crossentropy(y_dis, y) > Flux.crossentropy(y_dis, y_smoothed)
true
```
"""
function label_smoothing(y::Union{AbstractArray,Number}, α::Number; dims::Int = 1)
    if !(0 < α < 1)
        throw(ArgumentError("α must be between 0 and 1"))
    end
    if dims == 0
        y_smoothed = y .* (1 - α) .+ α*1//2
    elseif dims == 1
        y_smoothed = y .* (1 - α) .+ α* 1 // size(y, 1)
    else
        throw(ArgumentError("`dims` should be either 0 or 1"))
    end
    return y_smoothed
end

"""
    crossentropy(ŷ, y; dims = 1, ϵ = eps(ŷ), agg = mean)

Return the cross entropy between the given probability distributions;
calculated as

    agg(-sum(y .* log.(ŷ .+ ϵ); dims))

Cross entropy is typically used as a loss in multi-class classification,
in which case the labels `y` are given in a one-hot format.
`dims` specifies the dimension (or the dimensions) containing the class probabilities.
The prediction `ŷ` is supposed to sum to one across `dims`,
as would be the case with the output of a [`softmax`](@ref) operation.

For numerical stability, it is recommended to use [`logitcrossentropy`](@ref)
rather than `softmax` followed by `crossentropy` .

Use [`label_smoothing`](@ref) to smooth the true labels as preprocessing before
computing the loss.

See also: [`logitcrossentropy`](@ref), [`binarycrossentropy`](@ref), [`logitbinarycrossentropy`](@ref).

# Example
```jldoctest
julia> y_label = Flux.onehotbatch([0, 1, 2, 1, 0], 0:2)
3×5 Flux.OneHotArray{UInt32,3,1,2,Array{UInt32,1}}:
 1  0  0  0  1
 0  1  0  1  0
 0  0  1  0  0

julia> y_model = softmax(reshape(-7:7, 3, 5) .* 1f0)
3×5 Array{Float32,2}:
 0.0900306  0.0900306  0.0900306  0.0900306  0.0900306
 0.244728   0.244728   0.244728   0.244728   0.244728
 0.665241   0.665241   0.665241   0.665241   0.665241

julia> sum(y_model; dims=1)
1×5 Array{Float32,2}:
 1.0  1.0  1.0  1.0  1.0

julia> Flux.crossentropy(y_model, y_label)
1.6076053f0

julia> 5 * ans ≈ Flux.crossentropy(y_model, y_label; agg=sum)
true

julia> y_smooth = Flux.label_smoothing(y_label, 0.15f0)
3×5 Array{Float32,2}:
 0.9   0.05  0.05  0.05  0.9
 0.05  0.9   0.05  0.9   0.05
 0.05  0.05  0.9   0.05  0.05

julia> Flux.crossentropy(y_model, y_smooth)
1.5776052f0
```
"""
function crossentropy(ŷ, y; dims = 1, agg = mean, ϵ = epseltype(ŷ))
    agg(.-sum(xlogy.(y, ŷ .+ ϵ); dims = dims))
end

"""
    logitcrossentropy(ŷ, y; dims = 1, agg = mean)

Return the cross entropy calculated by

    agg(-sum(y .* logsoftmax(ŷ; dims); dims))

This is mathematically equivalent to `crossentropy(softmax(ŷ), y)`,
but is more numerically stable than using functions [`crossentropy`](@ref)
and [`softmax`](@ref) separately.

See also: [`binarycrossentropy`](@ref), [`logitbinarycrossentropy`](@ref), [`label_smoothing`](@ref).

# Example
```jldoctest
julia> y_label = Flux.onehotbatch(collect("abcabaa"), 'a':'c')
3×7 Flux.OneHotArray{UInt32,3,1,2,Array{UInt32,1}}:
 1  0  0  1  0  1  1
 0  1  0  0  1  0  0
 0  0  1  0  0  0  0

julia> y_model = reshape(vcat(-9:0, 0:9, 7.5f0), 3, 7)
3×7 Array{Float32,2}:
 -9.0  -6.0  -3.0  0.0  2.0  5.0  8.0
 -8.0  -5.0  -2.0  0.0  3.0  6.0  9.0
 -7.0  -4.0  -1.0  1.0  4.0  7.0  7.5

julia> Flux.logitcrossentropy(y_model, y_label)
1.5791205f0

julia> Flux.crossentropy(softmax(y_model), y_label)
1.5791197f0
```
"""
function logitcrossentropy(ŷ, y; dims = 1, agg = mean)
    agg(.-sum(y .* logsoftmax(ŷ; dims = dims); dims = dims))
end

"""
    binarycrossentropy(ŷ, y; agg = mean, ϵ = eps(ŷ))

Return the binary cross-entropy loss, computed as

    agg(@.(-y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)))

Where typically, the prediction `ŷ` is given by the output of a [`sigmoid`](@ref) activation.
The `ϵ` term is included to avoid infinity. Using [`logitbinarycrossentropy`](@ref) is recomended
over `binarycrossentropy` for numerical stability.

Use [`label_smoothing`](@ref) to smooth the `y` value as preprocessing before
computing the loss.

See also: [`crossentropy`](@ref), [`logitcrossentropy`](@ref).

# Examples
```jldoctest
julia> y_bin = Bool[1,0,1]
3-element Array{Bool,1}:
 1
 0
 1

julia> y_prob = softmax(reshape(vcat(1:3, 3:5), 2, 3) .* 1f0)
2×3 Array{Float32,2}:
 0.268941  0.5  0.268941
 0.731059  0.5  0.731059

julia> Flux.binarycrossentropy(y_prob[2,:], y_bin)
0.43989f0

julia> all(p -> 0<p<1, y_prob[2,:])  # else DomainError
true

julia> y_hot = Flux.onehotbatch(y_bin, 0:1)
2×3 Flux.OneHotArray{UInt32,2,1,2,Array{UInt32,1}}:
 0  1  0
 1  0  1

julia> Flux.crossentropy(y_prob, y_hot)
0.43989f0
```
"""
function binarycrossentropy(ŷ, y; agg = mean, ϵ = epseltype(ŷ))
    agg(@.(-xlogy(y, ŷ + ϵ) - xlogy(1 - y, 1 - ŷ + ϵ)))
end
# Re-definition to fix interaction with CuArrays.
# CUDA.@cufunc binarycrossentropy(ŷ, y; ϵ = eps(ŷ)) = -y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)

"""
    logitbinarycrossentropy(ŷ, y; agg = mean)

Mathematically equivalent to
[`binarycrossentropy(σ(ŷ), y)`](@ref) but is more numerically stable.

See also: [`crossentropy`](@ref), [`logitcrossentropy`](@ref).

# Examples
```jldoctest
julia> y_bin = Bool[1,0,1];

julia> y_model = Float32[2, -1, pi]
3-element Array{Float32,1}:
  2.0
 -1.0
  3.1415927

julia> Flux.logitbinarycrossentropy(y_model, y_bin)
0.160832f0

julia> Flux.binarycrossentropy(sigmoid.(y_model), y_bin)
0.16083185f0
```
"""
function logitbinarycrossentropy(ŷ, y; agg = mean)
    agg(@.((1 - y) * ŷ - logσ(ŷ)))
end
# Re-definition to fix interaction with CuArrays.
# CUDA.@cufunc logitbinarycrossentropy(ŷ, y) = (1 - y) * ŷ - logσ(ŷ)


"""
    kldivergence(ŷ, y; agg = mean, ϵ = eps(ŷ))

Return the
[Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
between the given probability distributions.

The KL divergence is a measure of how much one probability distribution is different
from the other. It is always non-negative, and zero only when both the distributions are equal.

# Example
```jldoctest
julia> p1 = [1 0; 0 1]
2×2 Array{Int64,2}:
 1  0
 0  1

julia> p2 = fill(0.5, 2, 2)
2×2 Array{Float64,2}:
 0.5  0.5
 0.5  0.5

julia> Flux.kldivergence(p2, p1) ≈ log(2)
true

julia> Flux.kldivergence(p2, p1; agg = sum) ≈ 2log(2)
true

julia> Flux.kldivergence(p2, p2; ϵ = 0)  # about -2e-16 with the regulator
0.0

julia> Flux.kldivergence(p1, p2; ϵ = 0)  # about 17.3 with the regulator
Inf
```
"""
function kldivergence(ŷ, y; dims = 1, agg = mean, ϵ = epseltype(ŷ))
  entropy = agg(sum(xlogx.(y), dims = dims))
  cross_entropy = crossentropy(ŷ, y; dims = dims, agg = agg, ϵ = ϵ)
  return entropy + cross_entropy
end

"""
    poisson_loss(ŷ, y)

# Return how much the predicted distribution `ŷ` diverges from the expected Poisson
# distribution `y`; calculated as `sum(ŷ .- y .* log.(ŷ)) / size(y, 2)`.

[More information.](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson).
"""
poisson_loss(ŷ, y; agg = mean) = agg(ŷ .- xlogy.(y, ŷ))

"""
    hinge_loss(ŷ, y; agg = mean)

Return the [hinge_loss loss](https://en.wikipedia.org/wiki/Hinge_loss) given the
prediction `ŷ` and true labels `y` (containing 1 or -1); calculated as
`sum(max.(0, 1 .- ŷ .* y)) / size(y, 2)`.

See also: [`squared_hinge_loss`](@ref)
"""
hinge_loss(ŷ, y; agg = mean) =
  agg(max.(0, 1 .- ŷ .* y))

"""
    squared_hinge_loss(ŷ, y)

Return the squared hinge_loss loss given the prediction `ŷ` and true labels `y`
(containing 1 or -1); calculated as `sum((max.(0, 1 .- ŷ .* y)).^2) / size(y, 2)`.

See also: [`hinge_loss`](@ref)
"""
squared_hinge_loss(ŷ, y; agg = mean) =
  agg((max.(0, 1 .- ŷ .* y)) .^ 2)

"""
    dice_coeff_loss(ŷ, y; smooth = 1)

Return a loss based on the dice coefficient.
Used in the [V-Net](https://arxiv.org/abs/1606.04797) image segmentation
architecture.
Similar to the F1_score. Calculated as:

    1 - 2*sum(|ŷ .* y| + smooth) / (sum(ŷ.^2) + sum(y.^2) + smooth)
"""
dice_coeff_loss(ŷ, y; smooth = ofeltype(ŷ, 1.0)) =
  1 - (2 * sum(y .* ŷ) + smooth) / (sum(y .^ 2) + sum(ŷ .^ 2) + smooth) #TODO agg

"""
    tversky_loss(ŷ, y; β = 0.7)

Return the [Tversky loss](https://arxiv.org/abs/1706.05721).
Used with imbalanced data to give more weight to false negatives.
Larger β weigh recall more than precision (by placing more emphasis on false negatives)
Calculated as:
    1 - sum(|y .* ŷ| + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1)
"""
function tversky_loss(ŷ, y; β = ofeltype(ŷ, 0.7))
    #TODO add agg
    num = sum(y .* ŷ) + 1
    den = sum(y .* ŷ + β * (1 .- y) .* ŷ + (1 - β) * y .* (1 .- ŷ)) + 1
    1 - num / den
end

"""
    focal_loss(yhat, y; dims=1, agg=mean, gamma=2.0, eps = eps(eltype(yhat))

         𝛄: modulating factor.

Return the [focal_loss](https://arxiv.org/pdf/1708.02002.pdf)
Extremely useful for classification when you have highly imbalanced classes. It down-weights
well-classified examples and focuses on hard examples. Loss is much high for misclassified points as compared to well-classified points. Used in single-shot detectors.
"""
function focal_loss(ŷ, y; dims=1, agg=mean, 𝛄=2.0, eps = eps(eltype(ŷ)))
    ŷ = ŷ .+ eps
    p_t = [y==1 ? ŷ : 1-ŷ for (ŷ, y) in zip(ŷ, y)]
    ce = -log.(p_t)
    weight = (1 .- p_t) .^ 𝛄
    loss = weight .* ce
    agg(sum(loss, dims=dims))
end

```@meta
DocTestFilters = nothing
```
