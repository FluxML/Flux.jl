# Cost functions

mse(ŷ, y) = sum((ŷ .- y).^2)/length(y)

crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat) =
  -sum(y .* log.(ŷ)) / size(y, 2)

@deprecate logloss(x, y) crossentropy(x, y)

function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat)
  logŷ = logŷ .- maximum(logŷ, 1)
  ypred = logŷ .- log.(sum(exp.(logŷ), 1))
  -sum(y .* ypred) / size(y, 2)
end

"""
    layernormalization(α=1.0, β=0.0)

Creates a normalization layer based on https://arxiv.org/pdf/1607.06450.pdf

The differences are:

1) std here divides by N-1 (as does std in Julia) vs the paper N
2) this layer α and β are constant numbers (i.e. not learnable vectors)

To achieve the same effect of learnable vectors α and β oe can use
the ElementwiseLinear layer
"""
function layernormalization(α=1.0, β=0.0)
  function layer(y)
    _mean = mean(y)
    _std = sqrt.(sum((y.-_mean).^2) ./ (length(y)-1))
    _std /= α
    _mean -= β*_std
    return (y .- _mean) ./ _std
  end
end
