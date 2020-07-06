"""
xlogx(x)

Return `x * log(x)` for `x ≥ 0`, handling `x = 0` by taking the downward limit.
"""
function xlogx(x)
result = x * log(x)
ifelse(iszero(x), zero(result), result)
end

CUDA.@cufunc function xlogx(x)
result = x * log(x)
ifelse(iszero(x), zero(result), result)
end

"""
xlogy(x, y)

Return `x * log(y)` for `y > 0` with correct limit at `x = 0`.
"""
function xlogy(x, y)
result = x * log(y)
ifelse(iszero(x), zero(result), result)
end

CUDA.@cufunc function xlogy(x, y)
result = x * log(y)
ifelse(iszero(x), zero(result), result)
end

@adjoint function broadcasted(::typeof(xlogy), x::Zygote.Numeric, y::Zygote.Numeric)
res = xlogy.(x, y)
res, Δ -> (nothing, Zygote.unbroadcast(x, xlogy.(Δ, y)), Zygote.unbroadcast(y, Δ .* x ./ y))
end
