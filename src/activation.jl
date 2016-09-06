export σ, relu, softmax, flatten

σ(x) = 1 ./ (1 .+ exp.(-x))

back!(::typeof(σ), Δ, x) = Δ .* σ(x)./(1.-σ(x))

relu(x) = max(0, x)

back!(::typeof(relu), Δ, x) = Δ .* (x .< 0)

softmax(xs) = exp.(xs) ./ sum(exp.(xs))

flatten(xs) = reshape(xs, length(xs))

shape(::typeof(flatten), in) = prod(in)
