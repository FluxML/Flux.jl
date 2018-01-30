using NNlib: log_fast, logsoftmax

# Cost functions

mse(ŷ, y) = sum((ŷ .- y).^2)/length(y)

function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* log_fast.(ŷ) .* weight) / size(y, 2)
end

@deprecate logloss(x, y) crossentropy(x, y)

"""
    logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)

`logitcrossentropy` takes the logits of the output probability distribution `logŷ` and
the target probability distribution `y` as the inputs to compute the cross entropy loss.
It is mathematically equivalent to the combination of `softmax(logŷ)` and `crossentropy`,
i.e., `crossentropy(softmax(logŷ), y)`, but it is more numerically stable than the former.

    julia> srand(123);
    julia> x = randn(5, 4);
    julia> y = rand(10, 4);
    julia> y = y ./ sum(y, 1);
    julia> m = Dense(5, 10);
    julia> logŷ = m(x);
    julia> Flux.logitcrossentropy(logŷ, y)
    Tracked 0-dimensional Array{Float64,0}:
    2.44887
"""
function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* logsoftmax(logŷ) .* weight) / size(y, 2)
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
