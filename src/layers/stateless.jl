# Activation Functions

σ(x) = 1 / (1 + exp(-x))
# back!(::typeof(σ), Δ, x) = Δ .* σ(x).*(1.-σ(x))

relu(x) = max(0, x)
# back!(::typeof(relu), Δ, x) = Δ .* (x .> 0)

softmax(xs) = exp.(xs) ./ sum(exp.(xs), 2)

# Cost functions

mse(ŷ, y) = sumabs2(ŷ .- y)/2
# back!(::typeof(mse), Δ, ŷ, y) = Δ .* (ŷ .- y)

logloss(ŷ, y) = -sum(y .* log.(ŷ))
# back!(::typeof(logloss), Δ, ŷ, y) = 0 .- Δ .* y ./ ŷ
