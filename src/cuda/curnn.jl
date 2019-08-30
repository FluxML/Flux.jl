import ..Flux: Flux, relu
using CuArrays.CUDAnative
using CuArrays: @cuindex, cudims
using LinearAlgebra

function LinearAlgebra.copy_transpose!(dst::CuArray, src::CuArray)
  function kernel(dst, src)
    I = @cuindex dst
    dst[I...] = src[reverse(I)...]
    return
  end
  blk, thr = cudims(dst)
  @cuda blocks=blk threads=thr kernel(dst, src)
  return dst
end

CuRNN{T} = Flux.RNNCell{<:Union{typeof(tanh),typeof(relu)},<:CuArray{T,2},<:CuArray{T,1}}
CuGRU{T} = Flux.GRUCell{<:CuArray{T,2},<:CuArray{T,1}}
CuLSTM{T} = Flux.LSTMCell{<:CuArray{T,2},<:CuArray{T,1}}
CuRNNs{T} = Union{CuRNN{T},CuGRU{T},CuLSTM{T}}

function copyparams!(m::CuRNNs, d::CUDNN.RNNDesc)
  Wi, Wh = d.weights
  copy_transpose!(Wi, m.Wi)
  copy_transpose!(Wh, m.Wh)
  copy_transpose!(d.bias, m.b)
  return
end

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
  copyparams!(rnn, d)
  return d
end

using ..Flux: @adjoint

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

for RNN in (CuRNN, CuGRU)
  @eval @adjoint function (m::$RNN{T})(h::CuArray{T}, x::CuArray{T}) where T <: Union{Float32,Float64}
    reserve, (y, ho) = CUDNN.forwardTrain(desc(m), x, h)
    (ho, y), function (Δ)
      dho, dy = Δ
      h_ = CUDNN.hBatch(x, h)
      dx, dh = CUDNN.backwardData(descs[m], y, dy, dho, h_, reserve)
      (dWi, dWh), db = CUDNN.backwardWeights(descs[m], x, h_, y, reserve)
      dm = Ref{Any}((σ=nothing,Wi=transpose(dWi),Wh=transpose(dWh),b=db,h=nothing))
      (dm, unbroadcast(h, dh), dx)
    end
  end
end

@adjoint function (m::CuLSTM)((h, c)::Tuple{CuArray{T},CuArray{T}}, x::CuArray{T}) where T <: Union{Float32,Float64}
  reserve, (y, ho, co) = CUDNN.forwardTrain(desc(m), x, h, c)
  ((ho, co), y), function (Δ)
    dhc, dy = Δ
    dho, dco = dhc === nothing ? (nothing, nothing) : dhc
    h_ = CUDNN.hBatch(x, h)
    c_ = CUDNN.hBatch(x, c)
    dx, dh, dc = CUDNN.backwardData(descs[m], y, dy, dho, dco, h_, c_, reserve)
    (dWi, dWh), db = CUDNN.backwardWeights(descs[m], x, h_, y, reserve)
    dm = Ref{Any}((Wi=transpose(dWi),Wh=transpose(dWh),b=db,h=nothing,c=nothing))
    (dm, (unbroadcast(h, dh), unbroadcast(c, dc)), dx)
  end
end
