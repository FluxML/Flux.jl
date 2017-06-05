export σ, relu, softmax

# Sigmoid
σ(x) = 1 ./ (1 + exp.(-x))
back!(::typeof(σ), Δ, x) = Δ .* σ(x).*(1.-σ(x))

# Rectified Linear Unit
relu(x) = max(0, x)
back!(::typeof(relu), Δ, x) = Δ .* (x .> 0)

softmax(xs) = exp.(xs) ./ sum(exp.(xs), 2)

flatten(xs) = reshape(xs, size(xs, 1), :)

infer(::typeof(softmax), x) = x
infer(::typeof(tanh), x) = x
infer(::typeof(relu), x) = x
infer(::typeof(σ), x) = x

infer(::typeof(flatten), x::Dims) = (x[1], prod(x[2:end])...)
