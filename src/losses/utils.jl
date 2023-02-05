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

# Gaussian kernel std=1.5, length=11
const SSIM_KERNEL = 
    [0.00102838008447911,
    0.007598758135239185,
    0.03600077212843083,
    0.10936068950970002,
    0.2130055377112537,
    0.26601172486179436,
    0.2130055377112537,
    0.10936068950970002,
    0.03600077212843083,
    0.007598758135239185,
    0.00102838008447911]

"""
    ssim_kernel(T, N)

Return Gaussian kernel with σ=1.5 and side-length 11 for use in [`ssim`](@ref).
"""
function ssim_kernel(T::Type, N::Integer)
    if N-2 == 1
        kernel = SSIM_KERNEL
    elseif N-2 == 2
        kernel = SSIM_KERNEL*SSIM_KERNEL' 
    elseif N-2 == 3
        ks = length(SSIM_KERNEL)
        kernel = reshape(SSIM_KERNEL*SSIM_KERNEL', 1, ks, ks).*SSIM_KERNEL
    else
        throw("SSIM is only implemented for 3D/4D/5D inputs, dimension=$N provided.")
    end
    return reshape(T.(kernel), size(kernel)..., 1, 1)
end
ssim_kernel(x::Array{T, N}) where {T, N} = ssim_kernel(T, N)
ssim_kernel(x::AnyCuArray{T, N}) where {T, N} = cu(ssim_kernel(T, N))

