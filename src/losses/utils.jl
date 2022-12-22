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
ChainRulesCore.@scalar_rule xlogx(x) (log(y) + true)

function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
  for d in 1:max(ndims(ŷ), ndims(y)) 
   size(ŷ,d) == size(y,d) || throw(DimensionMismatch(
      "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
    ))
  end
end
_check_sizes(ŷ, y) = nothing  # pass-through, for constant label e.g. y = 1

ChainRulesCore.@non_differentiable _check_sizes(ŷ::Any, y::Any)

# Greek-letter keywords deprecated in Flux 0.13
# Arguments (old => new, :function, "β" => "beta")
function _greek_ascii_depwarn(βbeta::Pair, func = :loss, names = "" => "")
  Base.depwarn("""loss function $func no longer accepts greek-letter keyword $(names.first)
    please use ascii $(names.second) instead""", func)
  βbeta.first
end
_greek_ascii_depwarn(βbeta::Pair{Nothing}, _...) = βbeta.second
