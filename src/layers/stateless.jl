# Cost functions

mse(ŷ, y) = sum((ŷ .- y).^2)/length(y)
# back!(::typeof(mse), Δ, ŷ, y) = Δ .* (ŷ .- y)

logloss(ŷ, y) = -sum(y .* log.(ŷ)) / size(y, 2)
# back!(::typeof(logloss), Δ, ŷ, y) = 0 .- Δ .* y ./ ŷ
