using CuArrays.CUDNN: @check, libcudnn, cudnnStatus_t, cudnnTensorDescriptor_t,
  cudnnBatchNormMode_t, cudnnHandle_t, libcudnn_handle, cudnnDataType, TensorDesc, FilterDesc
using CuArrays
using Flux

mutable struct DropoutDesc
  ptr::Ptr{Void}
  states::CuVector{UInt8}
end

Base.unsafe_convert(::Type{Ptr{Void}}, dd::DropoutDesc) = dd.ptr

function DropoutDesc(ρ::Real; seed::Integer=0)
  d = [C_NULL]
  s = Csize_t[0]
  @check ccall((:cudnnCreateDropoutDescriptor,libcudnn), cudnnStatus_t, (Ptr{Ptr{Void}},), d)
  @check ccall((:cudnnDropoutGetStatesSize,libcudnn),cudnnStatus_t,(Ptr{Void},Ptr{Csize_t}),libcudnn_handle[],s)
  states = CuArray{UInt8}(s[]) # TODO: can we drop this when ρ=0?
  desc = DropoutDesc(d[], states)
  @check ccall((:cudnnSetDropoutDescriptor,libcudnn),cudnnStatus_t,(Ptr{Void},Ptr{Void},Cfloat,Ptr{Void},Csize_t,Culonglong),
    desc,libcudnn_handle[],ρ,states,length(states),seed)
  finalizer(desc, x ->
    @check ccall((:cudnnDestroyDropoutDescriptor,libcudnn),cudnnStatus_t,(Ptr{Void},),x))
  return desc
end

const BATCHNORM_SPATIAL = 1
const BATCHNORM_ACTIVATION = 0
const BATCHNORM_MIN_EPS = 1e-5

@inline _wsize(y) = ((1 for _=1:ndims(y)-2)..., size(y)[end-1], 1)

mutable struct bncache
  mean
  ivar
end

bncache() = bncache(nothing, nothing)

(BN::BatchNorm)(x::CuArray{T}; cache = nothing) where T<:Union{Float32, Float64} =
  BN.λ.(cudnnBNForward(BN.γ, BN.β, x, BN.μ, BN.σ, BN.momentum, cache = cache, eps = BN.ϵ, training = BN.active))

function cudnnBNForward(g, b, x, running_mean::CuArray{T},
                        running_var::CuArray{T}, momentum;
                        cache = nothing, alpha = T(1), beta = T(0),
                        eps = T(1e-5), training = true) where T<:Union{Float32, Float64}
  y = similar(x)
  cudnnBNForward!(y, data(g), data(b), data(x), running_mean, running_var, momentum, cache = cache,
      alpha = alpha, beta = beta, eps = eps, training = training)
  y
end

function cudnnBNForward!(y::CuArray{T}, g::CuArray{T}, b::CuArray{T}, x::CuArray{T},
                        running_mean::CuArray{T}, running_var::CuArray{T},
                        momentum; cache = nothing,
                        alpha = T(1), beta = T(0),
                        eps = T(1e-5), training = true) where T<:Union{Float32, Float64}
  dims = _wsize(x)
  if(eps < BATCHNORM_MIN_EPS)
    warn("eps ",eps," is too small for CuDNN so eps has been assigned the value ", BATCHNORM_MIN_EPS)
    eps = BATCHNORM_MIN_EPS
  end
  xd = TensorDesc(x)
  yd = TensorDesc(y)
  gd = TensorDesc(T, (1,1,length(g),1))

  if(training)

    if(cache !== nothing)
      mean = cu(zeros(T, dims...))
      ivar = cu(ones(T, dims...))
    else
      mean = C_NULL
      ivar = C_NULL
    end

    @check ccall((:cudnnBatchNormalizationForwardTraining, libcudnn), cudnnStatus_t,
                     (cudnnHandle_t,cudnnBatchNormMode_t,
                      Ptr{T}, Ptr{T},
                      Ptr{Void}, Ptr{T},
                      Ptr{Void}, Ptr{T},
                      Ptr{Void}, Ptr{T}, Ptr{T},
                      Cdouble, Ptr{T}, Ptr{T},
                      Cdouble, Ptr{T}, Ptr{T}),
                      libcudnn_handle[], BATCHNORM_SPATIAL,
                      Ref(T(alpha)), Ref(T(beta)),
                      xd, x,
                      yd, y,
                      gd, g, b,
                      momentum, running_mean, running_var,
                      eps, mean, ivar)

    if(cache !== nothing)
      cache.mean = mean
      cache.ivar = ivar
    end
  else
    @check ccall((:cudnnBatchNormalizationForwardInference, libcudnn), cudnnStatus_t,
                     (Ptr{cudnnHandle_t},cudnnBatchNormMode_t,
                      Ptr{T}, Ptr{T},
                      Ptr{Void}, Ptr{T},
                      Ptr{Void}, Ptr{T},
                      Ptr{Void}, Ptr{T}, Ptr{T},
                      Ptr{T}, Ptr{T},
                      Cdouble),
                      libcudnn_handle[], BATCHNORM_SPATIAL,
                      Ref(T(alpha)), Ref(T(beta)),
                      xd, x,
                      yd, y,
                      gd, g, b,
                      running_mean, running_var,
                      eps)
  end
end

function cudnnBNBackward!(dg::CuArray{T}, g::CuArray{T}, db::CuArray{T},
                         dx::CuArray{T}, x::CuArray{T}, dy::CuArray{T},
                         running_mean::CuArray{T}, running_var::CuArray{T},
                         momentum; training = true,
                         cache = nothing, eps = T(1e-5),
                         alpha = T(1), beta = T(0),
                         dalpha = T(1), dbeta = T(0)) where T<:Union{Float32, Float64}
    if(training)

        if cache !== nothing
            mean, ivar = cache.mean, cache.ivar
            cache_verbose && info("mean and ivar are fetched from the cache")
        else
            mean, ivar = C_NULL, C_NULL
        end

        @check ccall((:cudnnBatchNormalizationBackward, libcudnn), cudnnStatus_t,
                     (cudnnHandle_t,cudnnBatchNormMode_t,
                      Ptr{T}, Ptr{T},
                      Ptr{T}, Ptr{T},
                      Ptr{Void}, Ptr{T},
                      Ptr{Void}, Ptr{T},
                      Ptr{Void}, Ptr{T},
                      Ptr{Void}, Ptr{T}, Ptr{T}, Ptr{T},
                      Cdouble, Ptr{T}, Ptr{T}),
                      libcudnn_handle[], BATCHNORM_SPATIAL,
                      Ref(T(alpha)), Ref(T(beta)),
                      Ref(T(dalpha)), Ref(T(dbeta)),
                      TensorDesc(x), x,
                      TensorDesc(dy), dy,
                      TensorDesc(dx), dx,
                      TensorDesc(g), g, dg, db,
                      eps, mean, ivar)
    else
        ivar = 1 ./ sqrt.(running_var .+ eps)
        dx = dy .* g .* ivar
        dg = sum(dy .* (x .- running_mean) .* ivar, _reddims(dy))
        db = sum(dy, _reddims(dy))
    end
end
