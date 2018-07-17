using CuArrays.CUDNN: @check, libcudnn, cudnnStatus_t, cudnnTensorDescriptor_t,
  cudnnBatchNormMode_t, cudnnHandle_t, libcudnn_handle, cudnnDataType, TensorDesc, FilterDesc
import ..Flux: data

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

@inline _wsize(y) = (map(_ -> 1, size(y)[1:end-2])..., size(y)[end-1], 1)

@inline _reddims(y) = (collect(1:ndims(y)-2)..., ndims(y))

mutable struct BNCache
  mean
  ivar
end

BNCache() = BNCache(nothing, nothing)

# CuDNN supports only 4D and 5D Tensors for BatchNorm Operations
# so use the native julia code when doing batchnorm on a 2D Array

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
  if eps < BATCHNORM_MIN_EPS
    warn("eps ",eps," is too small for CuDNN so eps has been assigned the value ", BATCHNORM_MIN_EPS)
    eps = BATCHNORM_MIN_EPS
  end
  xd = TensorDesc(x)
  yd = TensorDesc(y)
  gd = TensorDesc(T, dims)

  if training

    if cache !== nothing
      mean = zeros(CuArray{T}, dims...)
      ivar = ones(CuArray{T}, dims...)
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

    if cache !== nothing
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

function ∇batchnorm(g::CuArray{T}, b::CuArray{T}, x::CuArray{T}, dy::CuArray{T},
                    running_mean::CuArray{T}, running_var::CuArray{T}, momentum;
                    cache = nothing, eps = T(1e-5), alpha = T(1),
                    beta = T(0), training = true) where T<:Union{Float32, Float64}
  dg = similar(g)
  db = similar(b)
  dx = similar(x)
  cudnnBNBackward!(dg, g, db, dx, x, dy, running_mean, running_var, T(momentum),
    training = training, cache = cache, eps = eps, alpha = alpha, beta = beta)
  (dx, db, dg)
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
      mean, ivar = C_NULL, C_NULL
    end

    if eps < BATCHNORM_MIN_EPS
      warn("eps ",eps," is too small for CuDNN so eps has been assigned the value ", BATCHNORM_MIN_EPS)
      eps = BATCHNORM_MIN_EPS
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
                  xd, x,
                  dyd, dy,
                  dxd, dx,
                  gd, g, dg, db,
                  eps, mean, ivar)
  else
    ivar = 1 ./ sqrt.(reshape(running_var, _wsize(x)) .+ eps)
    dx .= dy .* reshape(g, _wsize(x)) .* ivar
    dg .= squeeze(sum(dy .* (x .- reshape(running_mean, _wsize(x))) .* ivar, _reddims(dy)), (1,2,4))
    db .= squeeze(sum(dy, _reddims(dy)), (1,2,4))
  end
end

# Flux Interface

<<<<<<< HEAD
import ..Flux: Flux
import ..Tracker: track, back, @back, istracked, TrackedArray

(BN::Flux.BatchNorm)(x::Union{CuParam{T,4},CuParam{T,5}}, cache = nothing) where T<:Union{Float32, Float64} =
  batchnorm(BN.γ, BN.β, x, BN.μ, BN.σ, BN.momentum; cache = cache, alpha = 1, beta = 0, eps = BN.ϵ, training = BN.active)
=======
function desc(rnn)
  d = haskey(descs, rnn) ? descs[rnn] : (descs[rnn] = RNNDesc(rnn))
  copyparams!(rnn, d)
  return d
end

import Flux.Tracker
import Flux.Tracker: data, istracked, track, unbroadcast, @grad, nobacksies

istrain(m::CuRNNs, args...) = any(x -> x isa TrackedArray, (m.Wi, m.Wh, m.b, args...))

function (m::CuRNN{T})(h::CuParam{T}, x::CuParam{T}) where T <: Union{Float32,Float64}
  result = istrain(m, h, x) ?
    track(m, x, h, m.Wi, m.Wh, m.b) :
    forward(desc(m), x, h)
  return result[2], result[1]
end

function (m::CuGRU{T})(h::CuParam{T}, x::CuParam{T}) where T <: Union{Float32,Float64}
  result = istrain(m, h, x) ?
    track(m, x, h, m.Wi, m.Wh, m.b) :
    forward(desc(m), x, h)
  return result[2], result[1]
end

function (m::CuLSTM{T})(h::NTuple{2,CuParam{T}}, x::CuParam{T}) where T <: Union{Float32,Float64}
  result = istrain(m, h, x) ?
    track(m, x, h[1], h[2], m.Wi, m.Wh, m.b) :
    forward(desc(m), x, h[1], h[2])
  return (result[2], result[3]), result[1]
end
>>>>>>> 071dcdda879a74cfd3c1115ac2c92087b38d4ae9

_batchnorm(g, b, x, running_mean, running_var, momentum,
           cache, alpha, beta, eps, training) =
  batchnorm(g, b, x, running_mean, running_var, momentum, cache = cache,
            alpha = alpha, beta = beta, eps = eps, training = training)

<<<<<<< HEAD
batchnorm(g::TrackedArray, b::TrackedArray, x::TrackedArray, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum;
          cache = nothing, alpha = T(1), beta = T(0),
          eps = T(1e-5), training = true) where T<:Union{Float32, Float64} =
  track(_batchnorm, g, b, x, running_mean, running_var, momentum, cache, alpha, beta, eps, training)

batchnorm(g::TrackedArray, b::TrackedArray, x::CuArray{T}, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; cache = nothing, alpha = T(1), beta = T(0),
          eps = T(1e-5), training = true) where T<:Union{Float32, Float64} =
  track(_batchnorm, g, b, x, running_mean, running_var, momentum, cache, alpha, beta, eps, training)

function back(::typeof(_batchnorm), Δ, g, b, x, running_mean, running_var, momentum, cache, alpha, beta, eps, training)
  deriv_tup = ∇batchnorm(data(g), data(b), data(x), Δ, running_mean, running_var, momentum,
                         cache = cache, alpha = alpha, beta = beta, eps = eps, training = training)
  @back(x, deriv_tup[1])
  @back(b, deriv_tup[2])
  @back(g, deriv_tup[3])
=======
@grad function (m::Union{CuRNN,CuGRU})(x, h, Wi, Wh, b)
  reserve, result = forwardTrain(desc(m), data(x), data(h))
  result, function (Δ)
    y, ho = result
    dy, dho = Δ
    h_ = hBatch(x, data(h))
    dx, dh = backwardData(descs[m], y, dy, dho, h_, reserve)
    (dWi, dWh), db = backwardWeights(descs[m], data(x), h_, y, reserve)
    nobacksies(:RNN, (dx, unbroadcast(size(h), dh), dWi.', dWh.', db))
  end
end

@grad function (m::CuLSTM)(x, h, c, Wi, Wh, b)
  reserve, result = forwardTrain(desc(m), data.((x, h, c))...)
  result, function (Δ)
    y, ho = result
    dy, dho, dco = Δ
    h_ = hBatch(x, data(h))
    c_ = hBatch(x, data(c))
    dx, dh, dc = backwardData(descs[m], y, dy, dho, dco, h_, c_, reserve)
    (dWi, dWh), db = backwardWeights(descs[m], data(x), h_, y, reserve)
    nobacksies(:RNN,
      (dx, unbroadcast(size(h), dh), unbroadcast(size(c), dc),
       dWi.', dWh.', db))
  end
>>>>>>> 071dcdda879a74cfd3c1115ac2c92087b38d4ae9
end
