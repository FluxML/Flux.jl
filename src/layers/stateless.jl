using NNlib: logsoftmax, logσ, log_fast

# Cost functions

mse(ŷ, y) = sum((ŷ .- y).^2)/length(y)

function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return @fix -sum(y .* log.(ŷ) .* weight) / size(y, 2)
end

@deprecate logloss(x, y) crossentropy(x, y)

function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* logsoftmax(logŷ) .* weight) / size(y, 2)
end

"""

    binarycrossentropy(ŷ, y)

Cross entropy loss for binary classification. `average` averages across the
batch. Machine epsilon (meps) has been added to maintain stability, preventing
Inf or Nan. This may also be used as a scalar function.
"""

function binarycrossentropy(ŷ, y; average = true, meps = eltype(y) <: AbstractFloat ? eps(eltype(y)) : 1f-7)
  bce  = -sum(y .* log_fast.(ŷ + meps) + (1 .- y) .* log_fast.(1 - ŷ + meps))
  if (average)
    bce /= length(y)
  elseif (!(size(ŷ)==(1,) && size(y)==(1,)) && (typeof(ŷ)<:AbstractVecOrMat
    && typeof(y)<:AbstractVecOrMat))
    warn("`crossentropy` may be a better choice than `binarycrossentropy`",
      " with apparently multiclass data.")
  end
  return bce
end

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
  μ′ = mean(x, 1)
  σ′ = std(x, 1, mean = μ′)
  return (x .- μ′) ./ σ′
end
