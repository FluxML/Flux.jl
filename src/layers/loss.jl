# Cost functions
@deprecate crossentropy nll
@deprecate logitcrossentropy cross_entropy
@deprecate binarycrossentropy bce
@deprecate logitbinarycrossentropy bce_logit


"""
  mse(ŷ, y; average=true, reduce=true)

Compute Mean Squared Error.
"""
function mse(ŷ, y; average=true, reduce=true) 
  l = (ŷ .- y).^2
  reduce || return l
  average ? sum(l) / size(ŷ, ndims(ŷ)) : sum(l)
end


"""
  cross_entropy(ŷ::AbstractMatrix, y::AbstractVector; 
      weight=1, average=true, reduce=true)

Combines [`logsoftmax`](@ref) with [`nll`](@ref).
"""
function cross_entropy(ŷ::AbstractMatrix, y; 
      weight=1, average=true, reduce=true)
  nll(logsoftmax(ŷ), y; weight=weight, average=average, reduce=reduce)
end

"""
  nll(ŷ::AbstractMatrix, y; 
      weight=1, average=true, reduce=true)

Negative log-likelihood. For *vector* target `y`, 
assumes the element of `y` to be inetegers in 1:K, where `K=size(ŷ,1)`. 
For `y` of *matrix* type instead, assumes onehot representation. 
"""
function nll(ŷ::AbstractMatrix, y::AbstractVector; 
      weight=1, average=true, reduce=true)
  indxs = _nll_indices(ŷ, y)
  l = (-weight .* ŷ)[indxs]
  reduce || return l
  average ? sum(l) / size(ŷ, 2) : sum(l) 
end

function nll(ŷ::AbstractMatrix, y::AbstractMatrix; 
      weight=1, average=true, reduce=true)
  l = sum((@. -weight * ŷ * y), 1) |> vec
  reduce || return l
  average ? sum(l) / size(ŷ, 2) : sum(l) 
end

function _nll_indices(ŷ, y::AbstractVector{T}) where {T<:Integer} 
  n = length(y)
  indices = Vector{Int}(n)
  d1, d2 = size(ŷ)
  n != d2 && throw(DimensionMismatch())
  @inbounds for i=1:n
    indices[i] = (i-1)*d1 + y[i]
  end
  indices
end

"""
  bce(ŷ, y; average=true, reduce=true)

Binary cross entropy. Assumes `0 < ŷ < 1` 
(e.g. the output of a sigmoid function).
See also [`bce_logit`](@ref).
"""
function bce(ŷ, y; average=true, reduce=true)
  l = @. -y*log(ŷ) - (1 - y)*log(1 - ŷ)
  reduce || return l
  average ? sum(l) / size(ŷ, ndims(ŷ)) : sum(l) 
end

"""
  bce_logit(ŷ, y; average=true, reduce=true)

Binary cross entropy preceeded by sigmoid activation.
See also [`bce`](@ref).
"""
function bce_logit(ŷ, y; average=true, reduce=true)
  max_val = max.(-ŷ, 0)
  l = @. ŷ - ŷ * y + max_val + log(exp(-max_val) + exp(-ŷ - max_val))
  reduce || return l
  average ? sum(l) / size(ŷ, ndims(ŷ)) : sum(l) 
end
