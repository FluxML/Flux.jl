
"""
    xlogx(x)

Return `x * log(x)` for `x ≥ 0`, handling `x == 0` by taking the limit from above, to get zero.
"""
function xlogx(x)
  result = x * log(x)
  ifelse(iszero(x), zero(result), result)
end

"""
    xlogy(x, y)

Return `x * log(y)` for `y > 0`, and zero when `x == 0`.
"""
function xlogy(x, y)
  result = x * log(y)
  ifelse(iszero(x), zero(result), result)
end

@adjoint function broadcasted(::typeof(xlogy), x::Zygote.Numeric, y::Zygote.Numeric)
  res = xlogy.(x, y)
  res, Δ -> (nothing, Zygote.unbroadcast(x, xlogy.(Δ, y)), Zygote.unbroadcast(y, Δ .* x ./ y))
end

ChainRulesCore.@scalar_rule xlogy(x, y) (log(y), x/y)  # should help Diffractor's broadcasting
ChainRulesCore.@scalar_rule xlogx(x) (log(x) + true)

const LowPrecision = Union{Float16, BFloat16}

# Losses compute in Float32 for Float16/BFloat16 inputs, as low-precision sums stall or overflow.
# The pullback converts the gradient back, so that the model's backward pass stays in low precision.
_upcast(x) = x
_upcast(x::LowPrecision) = Float32(x)
_upcast(x::AbstractArray{<:LowPrecision}) = Float32.(x)

function ChainRulesCore.rrule(::typeof(_upcast), x::Union{LowPrecision, AbstractArray{<:LowPrecision}})
  T = eltype(x)
  _upcast_back(Δ) = (NoTangent(), _downcast(T, unthunk(Δ)))
  return _upcast(x), _upcast_back
end

_downcast(::Type{T}, Δ::Real) where {T} = T(Δ)
_downcast(::Type{BFloat16}, Δ::Real) = _to_bf16(Δ)
_downcast(::Type{T}, Δ::AbstractArray{<:Real}) where {T} = _downcast.(T, Δ)
_downcast(::Type, Δ) = Δ

# The eltype that `_upcast` converts to, for the defaults of keywords such as `eps`.
_upcasttype(::Type{T}) where {T} = T
_upcasttype(::Type{<:LowPrecision}) = Float32

function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
  for d in 1:max(ndims(ŷ), ndims(y)) 
   size(ŷ,d) == size(y,d) || throw(DimensionMismatch(
      "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
    ))
  end
end
_check_sizes(ŷ, y) = nothing  # pass-through, for constant label e.g. y = 1

ChainRulesCore.@non_differentiable _check_sizes(ŷ::Any, y::Any)

# Whether `y` holds class labels for `ŷ` along `dims` rather than having the size of `ŷ`: an
# integer array with dimension `dims` dropped, or a one-hot array along `dims == 1`.
_islabels(ŷ, y, dims) = false
_islabels(ŷ::AbstractArray, y::AbstractArray{<:Integer}, dims::Integer) = ndims(y) == ndims(ŷ) - 1
_islabels(ŷ::AbstractArray, y::OneHotArray{<:Any, <:Any, <:Any, <:AbstractArray}, dims::Integer) =
  dims == 1 && size(y) == size(ŷ)

# The entries `ŷ[.., y, ..]` picked along `dims` by the class labels `y`, with the size of `y`, or
# for a one-hot `y` the size of `ŷ` with `dims` reduced to 1, as when `y` is a dense array.
_gather_labels(ŷ::AbstractArray, y::AbstractArray, dims::Integer) =
  _gather(ŷ, _label_indices(ŷ, y, dims))

function _gather_labels(ŷ::AbstractArray, y::OneHotArray, dims::Integer)
  labels = _indices(y)
  return reshape(_gather_labels(ŷ, labels, dims), 1, size(labels)...)
end

# `x[idx]` for linear indices `idx`. Unlike that of `getindex`, the gradient of `NNlib.gather` stays
# on the GPU. Overloaded for traced arrays in FluxReactantExt.
_gather(x::AbstractArray, idx::AbstractArray) = NNlib.gather(vec(x), idx)

# Linear indices into `ŷ`, viewed as a `prod(size(ŷ)[1:dims-1]) × size(ŷ, dims) × :` array.
# The closures capture `sz`, not `ŷ`, so that Reactant doesn't trace these reductions.
function _label_indices(ŷ::AbstractArray, y::AbstractArray, dims::Integer)
  sz = size(ŷ)
  _check_labels(sz, y, dims)
  A = prod(d -> sz[d], 1:dims-1; init = 1)
  C = sz[dims]
  B = prod(d -> sz[d], dims+1:length(sz); init = 1)
  idx = (1:A) .+ A .* (reshape(y, A, B) .- 1) .+ (A * C) .* (0:B-1)'
  return reshape(idx, size(y))
end

function _check_labels(sz::Dims, y::AbstractArray, dims::Integer)
  1 <= dims <= length(sz) || throw(ArgumentError(
    "dims = $dims is not a dimension of ŷ, which has size $sz"))
  ysz = ntuple(d -> sz[d < dims ? d : d + 1], Val(ndims(y)))
  size(y) == ysz || throw(DimensionMismatch(
    "with class labels, the loss expects size(y) = $ysz, the size of ŷ = $sz without dimension \
     dims = $dims, got size(y) = $(size(y))"))
  _check_label_values(y, sz[dims])
end

# Overloaded for traced arrays in FluxReactantExt, whose values are unknown at trace time.
_check_label_values(y, C) = all(l -> 1 <= l <= C, y) || throw(ArgumentError(
  "class labels must be in 1:$C, got labels from $(minimum(y)) to $(maximum(y))"))

ChainRulesCore.@non_differentiable _islabels(::Any...)
ChainRulesCore.@non_differentiable _label_indices(::Any...)

# `logsumexp(ŷ; dims) .- ŷ[y]` for class labels `y`, with the size of `_gather_labels(ŷ, y, dims)`,
# in Float32 for low-precision `ŷ`. Its rule writes the cotangent
# `(softmax(ŷ; dims) .- onehot(y)) .* Δ` in one broadcast, in the eltype of `ŷ`.
function _logitce(ŷ::AbstractArray, y::AbstractArray, dims::Integer)
  ŷy = _upcast(_gather_labels(ŷ, y, dims))  # checks the labels before reducing ŷ
  return reshape(logsumexp(_upcast(ŷ); dims), size(ŷy)) .- ŷy
end

function ChainRulesCore.rrule(::typeof(_logitce), ŷ::AbstractArray, y::AbstractArray, dims::Integer)
  ŷy = _upcast(_gather_labels(ŷ, y, dims))
  lse = logsumexp(_upcast(ŷ); dims)
  function _logitce_back(Δ)
    class = reshape(axes(ŷ, dims), ntuple(d -> d == dims ? size(ŷ, d) : 1, Val(ndims(ŷ))))
    labels, Δ = reshape(_labels(y), size(lse)), reshape(unthunk(Δ), size(lse))
    return NoTangent(), _logitce_grad.(ŷ, lse, class .== labels, Δ), NoTangent(), NoTangent()
  end
  return reshape(lse, size(ŷy)) .- ŷy, _logitce_back
end

_logitce_grad(x::T, lse, isy, Δ) where {T} = _downcast(float(T), (exp(_upcast(x) - lse) - isy) * Δ)

_labels(y::AbstractArray) = y
_labels(y::OneHotArray) = _indices(y)
