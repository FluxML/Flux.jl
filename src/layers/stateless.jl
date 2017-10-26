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
