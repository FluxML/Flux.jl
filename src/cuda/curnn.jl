import ..Flux: Flux, relu

CuRNN{T} = Flux.RNNCell{<:Union{typeof(tanh),typeof(relu)},<:CuArray{T,2},<:CuArray{T,1}}
CuGRU{T} = Flux.GRUCell{<:CuArray{T,2},<:CuArray{T,1}}
CuLSTM{T} = Flux.LSTMCell{<:CuArray{T,2},<:CuArray{T,1}}
CuRNNs{T} = Union{CuRNN{T},CuGRU{T},CuLSTM{T}}

function CUDNN.RNNDesc(m::CuRNNs{T}) where T
  h, i = length(m.h), size(m.Wi, 2)
  mode = m isa CuRNN ?
    (m.σ == tanh ? CUDNN.CUDNN_RNN_TANH : CUDNN.CUDNN_RNN_RELU) :
    m isa CuGRU ? CUDNN.CUDNN_GRU : CUDNN.CUDNN_LSTM
  r = CUDNN.RNNDesc{T}(mode, i, h)
  return r
end

const descs = WeakKeyDict()

function desc(rnn)
  d = haskey(descs, rnn) ? descs[rnn] : (descs[rnn] = CUDNN.RNNDesc(rnn))
  CUDNN.setweights!(d, rnn.Wi, rnn.Wh, rnn.b)
  return d
end

import Zygote
using Zygote: @adjoint

function (m::CuRNN{T})(h::CuArray{T}, x::CuArray{T}) where T <: Union{Float32,Float64}
  y, h′ = CUDNN.forward(desc(m), x, h)
  return h′, y
end

function (m::CuGRU{T})(h::CuArray{T}, x::CuArray{T}) where T <: Union{Float32,Float64}
  y, h′ = CUDNN.forward(desc(m), x, h)
  return h′, y
end

function (m::CuLSTM{T})(h::NTuple{2,CuArray{T}}, x::CuArray{T}) where T <: Union{Float32,Float64}
  y, h′, c′ = CUDNN.forward(desc(m), x, h[1], h[2])
  return (h′, c′), y
end

(m::CuRNN{T})(h::CuArray{T}, x) where T <: Union{Float32,Float64} = m(h, CuArray{T}(x))
(m::CuGRU{T})(h::CuArray{T}, x) where T <: Union{Float32,Float64} = m(h, CuArray{T}(x))
(m::CuLSTM{T})(h::NTuple{2,CuArray{T}}, x) where T <: Union{Float32,Float64} = m(h, CuArray{T}(x))

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
  length(x) == length(Δ) ? trim(x, Δ) :
    trim(x, sum(Δ, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))))

coerce_cuda(x::Union{CuArray,Nothing}) = x
coerce_cuda(x::Tuple) = coerce_cuda.(x)

coerce_cuda(x::AbstractArray) = x .+ CUDA.fill(0)

function struct_grad!(cx::Zygote.Context, x, x̄)
  for f in fieldnames(typeof(x))
    Zygote.accum_param(cx, getfield(x, f), getfield(x̄, f))
  end
  dx = Zygote.grad_mut(cx, x)
  dx[] = Zygote.accum(dx[], x̄)
  return dx
end

for RNN in (CuRNN, CuGRU)
  @eval @adjoint function (m::$RNN{T})(h::CuArray{T}, x::CuArray{T}) where T <: Union{Float32,Float64}
    (y, ho), back = CUDNN.pullback(desc(m), x, h)
    (ho, y), function (Δ)
      dho, dy = coerce_cuda(Δ) # Support FillArrays etc.
      m̄ = back(dy, dho)
      dm = struct_grad!(__context__, m, (σ=nothing,Wi=transpose(m̄.Wi),Wh=transpose(m̄.Wh),b=m̄.b,h=nothing))
      (dm, unbroadcast(h, m̄.h), m̄.x)
    end
  end
end

@adjoint function (m::CuLSTM)((h, c)::Tuple{CuArray{T},CuArray{T}}, x::CuArray{T}) where T <: Union{Float32,Float64}
  (y, ho, co), back = CUDNN.pullback(desc(m), x, h, c)
  ((ho, co), y), function (Δ)
    dhc, dy = coerce_cuda(Δ) # Support FillArrays etc.
    dho, dco = dhc === nothing ? (nothing, nothing) : dhc
    m̄ = back(dy, dho, dco)
    dm = struct_grad!(__context__, m, (σ=nothing,Wi=transpose(m̄.Wi),Wh=transpose(m̄.Wh),b=m̄.b,h=nothing,c=nothing))
    (dm, (unbroadcast(h, m̄.h), unbroadcast(c, m̄.c)), m̄.x)
  end
end
