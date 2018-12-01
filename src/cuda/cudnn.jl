using .CuArrays.CUDNN: @check, libcudnn, cudnnStatus_t, cudnnTensorDescriptor_t,
  cudnnBatchNormMode_t, cudnnHandle_t, cudnnDataType, TensorDesc, FilterDesc,
  version, CUDNNFloat
import ..Flux: data
using LinearAlgebra

mutable struct DropoutDesc
  ptr::Ptr{Nothing}
  states::CuVector{UInt8}
end

Base.unsafe_convert(::Type{Ptr{Nothing}}, dd::DropoutDesc) = dd.ptr

function DropoutDesc(ρ::Real; seed::Integer=0)
  d = [C_NULL]
  s = Csize_t[0]
  @check ccall((:cudnnCreateDropoutDescriptor,libcudnn), cudnnStatus_t, (Ptr{Ptr{Nothing}},), d)
  @check ccall((:cudnnDropoutGetStatesSize,libcudnn),cudnnStatus_t,(Ptr{Nothing},Ptr{Csize_t}),handle(),s)
  states = CuArray{UInt8}(undef, s[]) # TODO: can we drop this when ρ=0?
  desc = DropoutDesc(d[], states)
  @check ccall((:cudnnSetDropoutDescriptor,libcudnn),cudnnStatus_t,(Ptr{Nothing},Ptr{Nothing},Cfloat,Ptr{Nothing},Csize_t,Culonglong),
    desc,handle(),ρ,states,length(states),seed)
  finalizer(desc) do x
    @check ccall((:cudnnDestroyDropoutDescriptor,libcudnn),cudnnStatus_t,(Ptr{Nothing},),x)
  end
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

# NOTE: CuDNN supports only 4D and 5D Tensors for BatchNorm Operations
# so reshape a 2D Tensor into 4D
batchnorm(g::CuArray{T}, b::CuArray{T}, x::CuArray{T, 2},
          running_mean::CuArray{T}, running_var::CuArray{T}, momentum;
          cache = nothing, alpha = T(1), beta = T(0),
          eps = T(1e-5), training = true) where T<:CUDNNFloat =
  dropdims(batchnorm(g, b, reshape(x, 1, 1, size(x, 1), size(x, 2)), running_mean, running_var, momentum,
            cache = cache, alpha = alpha, beta = beta, eps = eps, training = training), dims = (1, 2))

function batchnorm(g::CuArray{T}, b::CuArray{T}, x::Union{CuArray{T, 4},CuArray{T,5}},
                   running_mean::CuArray{T}, running_var::CuArray{T}, momentum;
                   cache = nothing, alpha = T(1), beta = T(0),
                   eps = T(1e-5), training = true) where T<:CUDNNFloat
  y = similar(x)
  cudnnBNForward!(y, g, b, x, running_mean, running_var, momentum, cache = cache,
      alpha = alpha, beta = beta, eps = eps, training = training)
  y
end

function cudnnBNForward!(y::CuArray{T}, g::CuArray{T}, b::CuArray{T}, x::CuArray{T},
                        running_mean::CuArray{T}, running_var::CuArray{T},
                        momentum; cache = nothing,
                        alpha = T(1), beta = T(0),
                        eps = T(1e-5), training = true) where T<:CUDNNFloat
  dims = _wsize(x)
  if eps < BATCHNORM_MIN_EPS
    # warn("eps ",eps," is too small for CuDNN so eps has been assigned the value ", BATCHNORM_MIN_EPS)
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
                  Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Ptr{T}, Ptr{T},
                  Cdouble, Ptr{T}, Ptr{T},
                  Cdouble, Ptr{T}, Ptr{T}),
                  handle(), BATCHNORM_SPATIAL,
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
                  Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Ptr{T}, Ptr{T},
                  Ptr{T}, Ptr{T},
                  Cdouble),
                  handle(), BATCHNORM_SPATIAL,
                  Ref(T(alpha)), Ref(T(beta)),
                  xd, x,
                  yd, y,
                  gd, g, b,
                  running_mean, running_var,
                  eps)
  end
end

function ∇batchnorm(g::CuArray{T}, b::CuArray{T}, x::CuArray{T, 2}, dy::CuArray{T, 2},
           running_mean::CuArray{T}, running_var::CuArray{T}, momentum;
           cache = nothing, eps = T(1e-5), alpha = T(1),
           beta = T(0), training = true) where T<:CUDNNFloat
  dg, db, dx = ∇batchnorm(g, b, reshape(x, 1, 1, size(x, 1), size(x, 2)), reshape(dy, 1, 1, size(dy, 1),
                          size(dy, 2)), running_mean, running_var, momentum, cache = cache, eps = eps,
                          alpha = alpha, beta = beta, training = training)
  (dg, db, dropdims(dx, dims = (1, 2)))
end

function ∇batchnorm(g::CuArray{T}, b::CuArray{T}, x::CuArray{T}, dy::CuArray{T},
                    running_mean::CuArray{T}, running_var::CuArray{T}, momentum;
                    cache = nothing, eps = T(1e-5), alpha = T(1),
                    beta = T(0), training = true) where T<:CUDNNFloat
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
                          dalpha = T(1), dbeta = T(0), training = true) where T<:CUDNNFloat
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
      eps = BATCHNORM_MIN_EPS
    end

    @check ccall((:cudnnBatchNormalizationBackward, libcudnn), cudnnStatus_t,
                 (cudnnHandle_t,cudnnBatchNormMode_t,
                  Ptr{T}, Ptr{T},
                  Ptr{T}, Ptr{T},
                  Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Ptr{T}, Ptr{T}, Ptr{T},
                  Cdouble, Ptr{T}, Ptr{T}),
                  handle(), BATCHNORM_SPATIAL,
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
    dg .= squeeze(sum(dy .* (x .- reshape(running_mean, _wsize(x))) .* ivar, _reddims(dy)), dims = (1,2,4))
    db .= squeeze(sum(dy, _reddims(dy)), dims = (1,2,4))
  end
end

# Flux Interface

import Flux.Tracker
import Flux.Tracker: stracked, track, unbroadcast, @grad, nobacksies
using NNlib: padtuple, cdims, dilation_dims, conv, ∇conv_data, ∇conv_filter
using CuArrays.CUDNN: conv_workspace, cudnnConvolutionBackwardBias, cudnnConvolutionBackwardData,
  cudnnConvolutionBackwardFilter, cudnnActivationBackward, cudnnAddTensor,
  cudnnGetConvolutionForwardWorkspaceSize, cudnnConvolutionBiasActivationForward

(BN::Flux.BatchNorm)(x::Union{CuParam{T,2},CuParam{T,4},CuParam{T,5}}, cache = nothing) where T<:CUDNNFloat =
  batchnorm(BN.γ, BN.β, x, BN.μ, BN.σ², BN.momentum; cache = cache, alpha = 1, beta = 0, eps = BN.ϵ, training = BN.active)

batchnorm(g::TrackedArray, b::TrackedArray, x::TrackedArray, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:CUDNNFloat =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::TrackedArray, b::TrackedArray, x::CuArray{T}, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:CUDNNFloat =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::TrackedArray, b::CuArray{T}, x::TrackedArray, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:CUDNNFloat =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::CuArray{T}, b::TrackedArray, x::CuArray{T}, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:CUDNNFloat =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::CuArray{T}, b::TrackedArray, x::TrackedArray, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:CUDNNFloat =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::TrackedArray, b::CuArray{T}, x::CuArray{T}, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:CUDNNFloat =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::CuArray{T}, b::CuArray{T}, x::TrackedArray, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:CUDNNFloat =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

@grad batchnorm(g, b, x, running_mean, running_var, momentum; kw...) =
  batchnorm(data.((g, b, x))..., running_mean, running_var, momentum; kw...), Δ -> (nobacksies(:batchnorm, ∇batchnorm(data.((g, b, x, Δ))..., running_mean, running_var, momentum; kw...))..., nothing, nothing, nothing)

function convbias!(y::CuArray{T}, x::CuArray{T}, w::CuArray{T}, b::CuArray{T};
                   pad = 0, stride = 1, flipkernel = 0, alpha = 1, dilation = 1,
                   workspace::Union{CuVector, Nothing}=nothing, algo=0,
                   activationMode = 5) where T<:CUDNNFloat
  if version() < v"6"
    all(x -> x == 1, dilation) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end
  if workspace === nothing
    workspace_size =
      cudnnGetConvolutionForwardWorkspaceSize(y, x, w, padding=pad, stride=stride, dilation=dilation,
                                              algo=algo, mode=flipkernel)
    workspace = workspace_size != 0 ? conv_workspace(workspace_size) : workspace
  else
    workspace_size = length(workspace[])
  end
  cudnnConvolutionBiasActivationForward(y, x, w, b, padding=pad, stride=stride, mode=flipkernel, alpha1=alpha, activationMode=activationMode, algo=algo, workspace=workspace, workspace_size=workspace_size)
end

function convbias(x::CuArray{T}, w::CuArray{T}, b::CuArray{T};
                  pad = 0, stride = 1, flipkernel = 0, alpha = 1, dilation = 1,
                  workspace::Union{CuVector, Nothing}=nothing, algo=0,
                  activationMode = 5) where T<:CUDNNFloat
  pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
  convbias!(similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_)),
            x, w, b, pad = pad_, stride = stride_, flipkernel = flipkernel, dilation = dilation,
            alpha = alpha, workspace = workspace, algo = algo, activationMode = activationMode)
end

∇conv_bias(Δ::CuArray{T}, b::CuArray{T}; pad = 0, beta = 0,
           stride = 1, mode = 0, alpha = 1, dilation = 1) where T<:CUDNNFloat =
  reshape(cudnnConvolutionBackwardBias(similar(b), Δ, alpha=alpha, beta=beta), :)

(m::Flux.Conv)(x::Union{CuParam{T,4},CuParam{T,5}})  where T<:CUDNNFloat =
  m.σ.(convbias(x, m.weight, m.bias, pad = m.pad, stride = m.stride, dilation = m.dilation))

convbias(x::TrackedArray, w::TrackedArray, b::TrackedArray; kw...) = track(convbias, x, w, b; kw...)

convbias(x::CuArray{T}, w::TrackedArray, b::TrackedArray; kw...) where T<:CUDNNFloat =
  track(convbias, x, w, b; kw...)

convbias(x::CuArray{T}, w::CuArray{T}, b::TrackedArray; kw...) where T<:CUDNNFloat =
  track(convbias, x, w, b; kw...)

convbias(x::TrackedArray, w::CuArray{T}, b::TrackedArray; kw...) where T<:CUDNNFloat =
  track(convbias, x, w, b; kw...)

convbias(x::TrackedArray, w::TrackedArray, b::CuArray{T}; kw...) where T<:CUDNNFloat =
  track(convbias, x, w, b; kw...)

convbias(x::CuArray{T}, w::TrackedArray, b::CuArray{T}; kw...) where T<:CUDNNFloat =
  track(convbias, x, w, b; kw...)

convbias(x::TrackedArray, w::CuArray{T}, b::CuArray{T}; kw...) where T<:CUDNNFloat =
  track(convbias, x, w, b; kw...)

convbias(x::TrackedArray, w::TrackedArray, b::CuArray{T}; kw...) where T<:CUDNNFloat =
  track(convbias, x, w, b; kw...)

@grad function convbias(x, w, b; kw...)
  bias = reshape(b, map(_->1, kw[2][2])..., :, 1)
  if version() >= v"7.1"
    y = convbias(data.((x, w, bias))...; kw...)
  else
    y = cudnnAddTensor(data(bias), conv(data.((x, w))...; kw...))
  end
  y, Δ -> (nobacksies(:convbias, ∇conv_data(data.((Δ, x, w))...; kw...), ∇conv_filter(data.((Δ, x, w))...; kw...), ∇conv_bias(data.((Δ, bias))...; kw...))
end
