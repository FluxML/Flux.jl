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

# This can be made an error in Flux v0.13, for now just a warning
function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
  for d in 1:max(ndims(ŷ), ndims(y)) 
    if size(ŷ,d) != size(y,d)
      @warn "size mismatch in loss function! In future this will be an error. In Flux <= 0.12 broadcasting acceps this, but may not give sensible results" summary(ŷ) summary(y) maxlog=3 _id=hash(size(y))
    end
  end
end
_check_sizes(ŷ, y) = nothing  # pass-through, for constant label e.g. y = 1

Zygote.@nograd _check_sizes
