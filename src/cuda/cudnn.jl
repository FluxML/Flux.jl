using CuArrays.CUDNN: handle, TensorDesc, FilterDesc

import CuArrays.CUDAdrv: CU_NULL

using LinearAlgebra

mutable struct DropoutDesc
  ptr::Ptr{Nothing}
  states::CuVector{UInt8}
end

Base.unsafe_convert(::Type{Ptr{Nothing}}, dd::DropoutDesc) = dd.ptr

function DropoutDesc(ρ::Real; seed::Integer=0)
  d = [C_NULL]
  s = Csize_t[0]
  CUDNN.cudnnCreateDropoutDescriptor(d)
  CUDNN.cudnnDropoutGetStatesSize(handle(), s)
  states = CuArray{UInt8}(undef, s[]) # TODO: can we drop this when ρ=0?
  desc = DropoutDesc(d[], states)
  CUDNN.cudnnSetDropoutDescriptor(desc, handle(), ρ, states, length(states), seed)
  finalizer(desc) do x
    CUDNN.cudnnDestroyDropoutDescriptor(x)
  end
  return desc
end

@inline _wsize(y) = (map(_ -> 1, size(y)[1:end-2])..., size(y)[end-1], 1)

@inline _reddims(y) = (collect(1:ndims(y)-2)..., ndims(y))

mutable struct BNCache
  mean
  ivar
end

BNCache() = BNCache(nothing, nothing)

# NOTE: CuDNN supports only 4D and 5D Tensors for BatchNorm Operations
# so reshape a 2D Tensor into 4D
batchnorm(g::CuArray{T}, b::CuArray{T}, x::CuArray{T, 2},
          running_mean::CuArray{T}, running_var::CuArray{T}, momentum;
          cache = nothing, alpha = T(1), beta = T(0),
          eps = T(1e-5), training = true) where T<:Union{Float32, Float64} =
  dropdims(batchnorm(g, b, reshape(x, 1, 1, size(x, 1), size(x, 2)), running_mean, running_var, momentum,
            cache = cache, alpha = alpha, beta = beta, eps = eps, training = training), dims = (1, 2))

function batchnorm(g::CuArray{T}, b::CuArray{T}, x::Union{CuArray{T, 4},CuArray{T,5}},
                   running_mean::CuArray{T}, running_var::CuArray{T}, momentum;
                   cache = nothing, alpha = T(1), beta = T(0),
                   eps = T(1e-5), training = true) where T<:Union{Float32, Float64}
  y = similar(x)
  cudnnBNForward!(y, g, b, x, running_mean, running_var, momentum, cache = cache,
      alpha = alpha, beta = beta, eps = eps, training = training)
  y
end

function cudnnBNForward!(y::CuArray{T}, g::CuArray{T}, b::CuArray{T}, x::CuArray{T},
                        running_mean::CuArray{T}, running_var::CuArray{T},
                        momentum; cache = nothing,
                        alpha = T(1), beta = T(0),
                        eps = T(1e-5), training = true) where T<:Union{Float32, Float64}
  dims = _wsize(x)
  if eps < CUDNN.CUDNN_BN_MIN_EPSILON
    # warn("eps ",eps," is too small for CuDNN so eps has been assigned the value ", CUDNN.CUDNN_BN_MIN_EPSILON)
    eps = CUDNN.CUDNN_BN_MIN_EPSILON
  end
  xd = TensorDesc(x)
  yd = TensorDesc(y)
  gd = TensorDesc(T, dims)

  if training

    if cache !== nothing
      mean = zeros(CuArray{T}, dims...)
      ivar = ones(CuArray{T}, dims...)
    else
      mean = CU_NULL
      ivar = CU_NULL
    end

    CUDNN.cudnnBatchNormalizationForwardTraining(handle(), CUDNN.CUDNN_BATCHNORM_SPATIAL, Ref(T(alpha)), Ref(T(beta)), xd, x, yd, y, gd, g, b, momentum, running_mean, running_var, eps, mean, ivar)

    if cache !== nothing
      cache.mean = mean
      cache.ivar = ivar
    end
  else
    CUDNN.cudnnBatchNormalizationForwardInference(handle(), CUDNN.CUDNN_BATCHNORM_SPATIAL, Ref(T(alpha)), Ref(T(beta)), xd, x, yd, y, gd, g, b, running_mean, running_var, eps)
  end
end

function ∇batchnorm(g::CuArray{T}, b::CuArray{T}, x::CuArray{T, 2}, dy::CuArray{T, 2},
           running_mean::CuArray{T}, running_var::CuArray{T}, momentum;
           cache = nothing, eps = T(1e-5), alpha = T(1),
           beta = T(0), training = true) where T<:Union{Float32, Float64}
  dg, db, dx = ∇batchnorm(g, b, reshape(x, 1, 1, size(x, 1), size(x, 2)), reshape(dy, 1, 1, size(dy, 1),
                          size(dy, 2)), running_mean, running_var, momentum, cache = cache, eps = eps,
                          alpha = alpha, beta = beta, training = training)
  (dg, db, dropdims(dx, dims = (1, 2)))
end

function ∇batchnorm(g::CuArray{T}, b::CuArray{T}, x::CuArray{T}, dy::CuArray{T},
                    running_mean::CuArray{T}, running_var::CuArray{T}, momentum;
                    cache = nothing, eps = T(1e-5), alpha = T(1),
                    beta = T(0), training = true) where T<:Union{Float32, Float64}
  dg = similar(g)
  db = similar(b)
  dx = similar(x)
  cudnnBNBackward!(dg, g, db, dx, x, dy, running_mean, running_var, T(momentum),
    training = training, cache = cache, eps = eps, alpha = alpha, beta = beta)
  (dg, db, dx)
end

function cudnnBNBackward!(dg::CuArray{T}, g::CuArray{T}, db::CuArray{T},
                          dx::CuArray{T}, x::CuArray{T}, dy::CuArray{T},
                          running_mean::CuArray{T}, running_var::CuArray{T},
                          momentum; cache = nothing, eps = T(1e-5),
                          alpha = T(1), beta = T(0),
                          dalpha = T(1), dbeta = T(0), training = true) where T<:Union{Float32, Float64}
  if training
    xd = TensorDesc(x)
    dyd = TensorDesc(dy)
    dxd = TensorDesc(dx)
    gd = TensorDesc(T, _wsize(x))
    if cache !== nothing
      mean, ivar = cache.mean, cache.ivar
      info("mean and ivar are fetched from the cache")
    else
      mean, ivar = CU_NULL, CU_NULL
    end

    if eps < CUDNN.CUDNN_BN_MIN_EPSILON
      eps = CUDNN.CUDNN_BN_MIN_EPSILON
    end

    CUDNN.cudnnBatchNormalizationBackward(handle(), CUDNN.CUDNN_BATCHNORM_SPATIAL, Ref(T(alpha)), Ref(T(beta)), Ref(T(dalpha)), Ref(T(dbeta)), xd, x, dyd, dy, dxd, dx, gd, g, dg, db, eps, mean, ivar)
  else
    ivar = 1 ./ sqrt.(reshape(running_var, _wsize(x)) .+ eps)
    dx .= dy .* reshape(g, _wsize(x)) .* ivar
    dg .= squeeze(sum(dy .* (x .- reshape(running_mean, _wsize(x))) .* ivar, _reddims(dy)), dims = (1,2,4))
    db .= squeeze(sum(dy, _reddims(dy)), dims = (1,2,4))
  end
end

# Flux Interface

(BN::Flux.BatchNorm)(x::Union{CuArray{T,2},CuArray{T,4},CuArray{T,5}}, cache = nothing) where T<:Union{Float32, Float64} =
  BN.λ.(batchnorm(BN.γ, BN.β, x, BN.μ, BN.σ², BN.momentum; cache = cache, alpha = 1, beta = 0, eps = BN.ϵ, training = Flux.istraining()))

@adjoint batchnorm(g, b, x, running_mean, running_var, momentum; kw...) =
  batchnorm(g, b, x, running_mean, running_var, momentum; kw...), Δ -> (∇batchnorm(g, b, x, Δ, running_mean, running_var, momentum; kw...)..., nothing, nothing, nothing)
