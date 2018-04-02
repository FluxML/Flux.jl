# Cost functions
function mse(ŷ, y; average=true, reduce=true) 
  l = (ŷ .- y).^2
  reduce || return l
  average ? sum(l) / size(ŷ, ndims(ŷ)) : sum(l)
end

function cross_entropy(ŷ::AbstractMatrix, y::AbstractVector; 
      weight=1, average=true, reduce=true)
  nll(logsoftmax(ŷ), y; weight=weight, average=average, reduce=reduce)
end
  
function nll(ŷ::AbstractMatrix, y::AbstractVector; 
      weight=1, average=true, reduce=true)
  indxs = _nll_indices(ŷ, y)
  l = (-weight .* ŷ)[indxs]
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

function bce(ŷ::AbstractArray, y::AbstractArray; 
    average=true, reduce=true)
  l = @. -y*log(ŷ) - (1 - y)*log(1 - ŷ)
  reduce || return l
  average ? sum(l) / size(ŷ, ndims(ŷ)) : sum(l) 
end

function bce_logit(ŷ::AbstractArray, y::AbstractArray; 
    average=true, reduce=true)
  max_val = max.(-ŷ, 0)
  l = @. ŷ - ŷ * y + max_val + log(exp(-max_val) + exp(-ŷ - max_val))
  reduce || return l
  average ? sum(l) / size(ŷ, ndims(ŷ)) : sum(l) 
end

"""
    normalise(x::AbstractVecOrMat)

Normalise each column of `x` to mean 0 and standard deviation 1.
"""
function normalise(x::AbstractVecOrMat)
  μ = mean(x, 1)
  σ = std(x, 1, mean = μ)
  return (x .- μ) ./ σ
end
