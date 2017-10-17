# Cost functions

mse(ŷ, y) = sum((ŷ .- y).^2)/length(y)

logloss(ŷ::AbstractVecOrMat, y::AbstractVecOrMat) =
  -sum(y .* log.(ŷ)) / size(y, 2)
