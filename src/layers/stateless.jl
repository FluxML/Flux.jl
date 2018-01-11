using NNlib: log_fast

# Cost functions

mse(ŷ, y) = sum((ŷ .- y).^2)/length(y)

function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* log_fast.(ŷ) .* weight) / size(y, 2)
end

@deprecate logloss(x, y) crossentropy(x, y)

function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat)
  logŷ = logŷ .- maximum(logŷ, 1)
  ypred = logŷ .- log_fast.(sum(exp.(logŷ), 1))
  -sum(y .* ypred) / size(y, 2)
end

function binarycrossentropy(ŷ, y; average=true)
  bce  = -sum(y .* log_fast.(ŷ) + (1 .- y) .* log_fast.(1 - ŷ))
  if (average)
    bce /= length(y)
  elseif !(size(ŷ )==(1,) && size(y)==(1,))
    warn("`crossentropy` may be a better choice than `binarycrossentropy` with apparently multiclass data.")
  end
  return bce
end


"""
    normalise(x::AbstractVecOrMat)

Normalise each column of `x` to mean 0 and standard deviation 1.
"""
function normalise(x::AbstractVecOrMat)
  μ′ = mean(x, 1)
  σ′ = std(x, 1, mean = μ′)
  return (x .- μ′) ./ σ′
end
