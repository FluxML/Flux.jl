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
crossentropy(logŷ, y) = crossentropy(logŷ, y, 1/size(y,2))
function crossentropy(logŷ, y, w) 
  logŷ = logŷ .-maximum(logŷ,1)
  ypred = logŷ .- log.( sum( exp.( logŷ),1))
  -sum(y .* w .* ypred)
end
