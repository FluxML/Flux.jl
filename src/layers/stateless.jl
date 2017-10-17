# Cost functions

mse(ŷ, y) = sum((ŷ .- y).^2)/length(y)
# back!(::typeof(mse), Δ, ŷ, y) = Δ .* (ŷ .- y)

logloss(ŷ, y) = -sum(y .* log.(ŷ)) / size(y, 2)
# back!(::typeof(logloss), Δ, ŷ, y) = 0 .- Δ .* y ./ ŷ

"""
    crossentropy(logŷ, y)
    crossentropy(logŷ,y,w)

    implement the cross-entropy loss function taking the logits (output before the softmax)
"""
crossentropy(logŷ, y::AbstractVector) = crossentropy(logŷ, onehotbatch(y,1:size(logŷ,1)), 1/length(y))
crossentropy(logŷ, y::AbstractMatrix) = crossentropy(logŷ, y, 1/size(y,2))
crossentropy(logŷ, y::AbstractVector, w) = crossentropy(logŷ, onehotbatch(y,1:size(logŷ,1)), w)
function crossentropy(logŷ, y::AbstractMatrix, w) 
  logŷ = logŷ .-maximum(logŷ,1)
  ypred = logŷ .- log.( sum( exp.( logŷ),1))
  -sum(y .* w .* ypred)
end
