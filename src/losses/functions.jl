"""
    mae(ŷ, y; agg=mean)

Return the loss corresponding to mean absolute error:

    agg(abs.(ŷ .- y))
"""
mae(ŷ, y; agg=mean) = agg(abs.(ŷ .- y))

"""
    mse(ŷ, y; agg=mean)

Return the loss corresponding to mean square error:

    agg((ŷ .- y).^2)
"""
mse(ŷ, y; agg=mean) = agg((ŷ .- y).^2)

"""
    msle(ŷ, y; agg=mean, ϵ=eps(ŷ))

The loss corresponding to mean squared logarithmic errors, calculated as

    agg((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2)

The `ϵ` term provides numerical stability.
Penalizes an under-estimation more than an over-estimatation.
"""
msle(ŷ, y; agg=mean, ϵ=epseltype(ŷ)) = agg((log.((ŷ .+ ϵ) ./ (y .+ ϵ))).^2)

"""
    huber_loss(ŷ, y; δ=1, agg=mean)

Return the mean of the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
given the prediction `ŷ` and true values `y`.

                 | 0.5 * |ŷ - y|^2,            for |ŷ - y| <= δ
    Huber loss = |
                 |  δ * (|ŷ - y| - 0.5 * δ), otherwise
"""
function huber_loss(ŷ, y; agg=mean, δ=ofeltype(ŷ, 1))
   abs_error = abs.(ŷ .- y)
   #TODO: remove dropgrad when Zygote can handle this function with CuArrays
   temp = Zygote.dropgrad(abs_error .<  δ)
   x = ofeltype(ŷ, 0.5)
   agg(((abs_error.^2) .* temp) .* x .+ δ*(abs_error .- x*δ) .* (1 .- temp))
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

Usage example:

    sf = 0.1
    y = onehotbatch([1, 1, 1, 0, 0], 0:1)
    y_smoothed = label_smoothing(ya, 2sf)
    y_sim = y .* (1-2sf) .+ sf
    y_dis = copy(y_sim)
    y_dis[1,:], y_dis[2,:] = y_dis[2,:], y_dis[1,:]
    @assert crossentropy(y_sim, y) < crossentropy(y_sim, y_smoothed)
    @assert crossentropy(y_dis, y) > crossentropy(y_dis, y_smoothed)
"""
function label_smoothing(y::Union{AbstractArray,Number}, α::Number; dims::Int=1)
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
    crossentropy(ŷ, y; dims=1, ϵ=eps(ŷ), agg=mean)

Return the cross entropy between the given probability distributions;
calculated as

    agg(-sum(y .* log.(ŷ .+ ϵ); dims=dims))

Cross entropy is typically used as a loss in multi-class classification,
in which case the labels `y` are given in a one-hot format.
`dims` specifies the dimension (or the dimensions) containing the class probabilities.
The prediction `ŷ` is supposed to sum to one across `dims`,
as would be the case with the output of a [`softmax`](@ref) operation.

Use [`label_smoothing`](@ref) to smooth the true labels as preprocessing before
computing the loss.

Use of [`logitcrossentropy`](@ref) is recomended over `crossentropy` for
numerical stability.

See also: [`logitcrossentropy`](@ref), [`binarycrossentropy`](@ref), [`logitbinarycrossentropy`](@ref), 
[`label_smoothing`](@ref)
"""
function crossentropy(ŷ, y; dims=1, agg=mean, ϵ=epseltype(ŷ))
    agg(.-sum(xlogy.(y, ŷ .+ ϵ); dims=dims))
end

"""
    logitcrossentropy(ŷ, y; dims=1, agg=mean)

Return the crossentropy computed after a [`logsoftmax`](@ref) operation;
calculated as

    agg(.-sum(y .* logsoftmax(ŷ; dims=dims); dims=dims))

Use [`label_smoothing`](@ref) to smooth the true labels as preprocessing before
computing the loss.

`logitcrossentropy(ŷ, y)` is mathematically equivalent to
[`crossentropy(softmax(ŷ), y)`](@ref) but it is more numerically stable.


See also: [`crossentropy`](@ref), [`binarycrossentropy`](@ref), [`logitbinarycrossentropy`](@ref), [`label_smoothing`](@ref)
"""
function logitcrossentropy(ŷ, y; dims=1, agg=mean)
    agg(.-sum(y .* logsoftmax(ŷ; dims=dims); dims=dims))
end

"""
    binarycrossentropy(ŷ, y; agg=mean, ϵ=eps(ŷ))

Return the binary cross-entropy loss, computed as

    agg(@.(-y*log(ŷ + ϵ) - (1-y)*log(1-ŷ + ϵ)))

The `ϵ` term provides numerical stability.

Typically, the prediction `ŷ` is given by the output of a [`sigmoid`](@ref) activation.

Use [`label_smoothing`](@ref) to smooth the `y` value as preprocessing before
computing the loss.

Use of `logitbinarycrossentropy` is recomended over `binarycrossentropy` for numerical stability.

See also: [`crossentropy`](@ref), [`logitcrossentropy`](@ref), [`logitbinarycrossentropy`](@ref), 
[`label_smoothing`](@ref)
"""
function binarycrossentropy(ŷ, y; agg=mean, ϵ=epseltype(ŷ))
    agg(@.(-xlogy(y, ŷ+ϵ) - xlogy(1-y, 1-ŷ+ϵ)))
end
# Re-definition to fix interaction with CuArrays.
# CUDA.@cufunc binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

"""
    logitbinarycrossentropy(ŷ, y; agg=mean)

Mathematically equivalent to
[`binarycrossentropy(σ(ŷ), y)`](@ref) but is more numerically stable.

Use [`label_smoothing`](@ref) to smooth the `y` value as preprocessing before
computing the loss.

See also: [`crossentropy`](@ref), [`logitcrossentropy`](@ref), [`binarycrossentropy`](@ref), [`label_smoothing`](@ref)
"""
function logitbinarycrossentropy(ŷ, y; agg=mean)
    agg(@.((1-y)*ŷ - logσ(ŷ)))
end
# Re-definition to fix interaction with CuArrays.
# CUDA.@cufunc logitbinarycrossentropy(ŷ, y) = (1 - y)*ŷ - logσ(ŷ)


"""
    kldivergence(ŷ, y; agg=mean)

Return the
[Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
between the given probability distributions.

KL divergence is a measure of how much one probability distribution is different
from the other.
It is always non-negative and zero only when both the distributions are equal
everywhere.
"""
function kldivergence(ŷ, y; dims=1, agg=mean, ϵ=epseltype(ŷ))
  entropy = agg(sum(xlogx.(y), dims=dims))
  cross_entropy = crossentropy(ŷ, y; dims=dims, agg=agg, ϵ=ϵ)
  return entropy + cross_entropy
end

"""
    poisson_loss(ŷ, y)

# Return how much the predicted distribution `ŷ` diverges from the expected Poisson
# distribution `y`; calculated as `sum(ŷ .- y .* log.(ŷ)) / size(y, 2)`.

[More information.](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson).
"""
poisson_loss(ŷ, y; agg=mean) = agg(ŷ .- xlogy.(y, ŷ))

"""
    hinge_loss(ŷ, y; agg=mean)

Return the [hinge_loss loss](https://en.wikipedia.org/wiki/Hinge_loss) given the
prediction `ŷ` and true labels `y` (containing 1 or -1); calculated as
`sum(max.(0, 1 .- ŷ .* y)) / size(y, 2)`.

See also: [`squared_hinge_loss`](@ref)
"""
hinge_loss(ŷ, y; agg=mean) = agg(max.(0, 1 .-  ŷ .* y))

"""
    squared_hinge_loss(ŷ, y)

Return the squared hinge_loss loss given the prediction `ŷ` and true labels `y`
(containing 1 or -1); calculated as `sum((max.(0, 1 .- ŷ .* y)).^2) / size(y, 2)`.

See also: [`hinge_loss`](@ref)
"""
squared_hinge_loss(ŷ, y; agg=mean) = agg((max.(0, 1 .- ŷ .* y)).^2)

"""
    dice_coeff_loss(ŷ, y; smooth=1)

Return a loss based on the dice coefficient.
Used in the [V-Net](https://arxiv.org/abs/1606.04797) image segmentation
architecture.
Similar to the F1_score. Calculated as:

    1 - 2*sum(|ŷ .* y| + smooth) / (sum(ŷ.^2) + sum(y.^2) + smooth)
"""
dice_coeff_loss(ŷ, y; smooth=ofeltype(ŷ, 1.0)) = 1 - (2*sum(y .* ŷ) + smooth) / (sum(y.^2) + sum(ŷ.^2) + smooth) #TODO agg

"""
    tversky_loss(ŷ, y; β=0.7)

Return the [Tversky loss](https://arxiv.org/abs/1706.05721).
Used with imbalanced data to give more weight to false negatives.
Larger β weigh recall more than precision (by placing more emphasis on false negatives)
Calculated as:
    1 - sum(|y .* ŷ| + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1)
"""
function tversky_loss(ŷ, y; β=ofeltype(ŷ, 0.7))
    #TODO add agg
    num = sum(y .* ŷ) + 1
    den = sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1
    1 - num / den
end
