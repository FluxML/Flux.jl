using CuArrays.CUDNN: @check, libcudnn, cudnnStatus_t, cudnnTensorDescriptor_t,
  cudnnBatchNormMode_t, cudnnHandle_t, cudnnDataType, TensorDesc, FilterDesc
using LinearAlgebra

const RNN_RELU = 0 # Stock RNN with ReLu activation
const RNN_TANH = 1 # Stock RNN with tanh activation
const LSTM = 2     # LSTM with no peephole connections
const GRU = 3      # Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1)

const LINEAR_INPUT = 0
const SKIP_INPUT = 1

const UNIDIRECTIONAL = 0
const BIDIRECTIONAL = 1

const RNN_ALGO_STANDARD = 0
const RNN_ALGO_PERSIST_STATIC = 1
const RNN_ALGO_PERSIST_DYNAMIC = 2

# param layout:
# RNN: [weight, bias] × [input, hidden]
# GRU: [weight, bias] × [input, hidden] × [reset, update, newmem]
# LSTM: [weight, bias] × [input, hidden] × [input, forget, newmem, output]

function params(w::CuVector, input, hidden, n = 1)
  slice(offset, shape) = reshape(view(w, offset.+(1:prod(shape))), shape)
  wx = slice(0, (input, hidden*n))
  wh = slice(length(wx), (hidden, hidden*n))
  bias = view(w, length(wx)+length(wh) .+ (1:hidden*n))
  (wx, wh), bias
end

mutable struct RNNDesc{T}
  mode::Int
  input::Int
  hidden::Int
  params::CuVector{T}
  weights::NTuple{2,CuMatrix{T}}
  bias::CuVector{T}
  ptr::Ptr{Nothing}
end

Base.unsafe_convert(::Type{Ptr{Nothing}}, d::RNNDesc) = d.ptr

function rnnParamSize(T, r, input)
  size = Csize_t[0]
  @check ccall((:cudnnGetRNNParamsSize, libcudnn), cudnnStatus_t, (Ptr{Nothing},Ptr{Nothing},Ptr{Nothing},Ptr{Csize_t},Cint),
    handle(), r, TensorDesc(T, (1,input,1)), size, cudnnDataType(T))
  return Int(size[])÷sizeof(T)
end

ngates(mode) = [1, 1, 4, 3][mode+1]
ngates(r::RNNDesc) = ngates(r.mode)

function RNNDesc{T}(mode::Int, input::Int, hidden::Int; layers = 1) where T
  d = [C_NULL]
  @check ccall((:cudnnCreateRNNDescriptor,libcudnn),cudnnStatus_t,(Ptr{Ptr{Nothing}},),d)

  dropoutDesc = DropoutDesc(0)
  inputMode = LINEAR_INPUT
  direction = UNIDIRECTIONAL
  algo = RNN_ALGO_STANDARD
  @check ccall((:cudnnSetRNNDescriptor_v6,libcudnn), cudnnStatus_t, (Ptr{Nothing},Ptr{Nothing},Cint,Cint,Ptr{Nothing},Cint,Cint,Cint,Cint,Cint),
    handle(),d[],hidden,layers,dropoutDesc,inputMode,direction,mode,algo,cudnnDataType(T))

  w = cuzeros(T, rnnParamSize(T, d[], input))
  # TODO: avoid reserve allocation here
  rd = RNNDesc{T}(mode, input, hidden, w, params(w, input, hidden, ngates(mode))..., d[])
  finalizer(rd) do x
    @check ccall((:cudnnDestroyRNNDescriptor,libcudnn),cudnnStatus_t,(Ptr{Nothing},),x)
  end
  return rd
end

function rnnWorkspaceSize(r::RNNDesc, seqlen, xdesc)
  size = Csize_t[0]
  @check ccall((:cudnnGetRNNWorkspaceSize, libcudnn), cudnnStatus_t, (Ptr{Nothing},Ptr{Nothing},Cint,Ptr{Ptr{Nothing}},Ptr{Csize_t}),
    handle(), r, seqlen, xdesc, size)
  return Int(size[])
end

const workspace = [CuVector{UInt8}(undef, 1)]

getworkspace(bytes) =
  length(workspace[]) ≥ bytes ?
    workspace[] :
    (workspace[] = CuVector{UInt8}(undef, bytes))

getworkspace(r::RNNDesc, seqlen, xdesc) =
  getworkspace(rnnWorkspaceSize(r, seqlen, xdesc))

function rnnTrainingReserveSize(r::RNNDesc, seqlen, xdesc)
  size = Csize_t[0]
  @check ccall((:cudnnGetRNNTrainingReserveSize,libcudnn), cudnnStatus_t, (Ptr{Nothing}, Ptr{Nothing}, Cint, Ptr{Ptr{Nothing}}, Ptr{Csize_t}),
    handle(), r, seqlen, xdesc, size)
  return Int(size[])
end

function cudnnRNNForward(rnn::RNNDesc{T}, seqlen, xd, x, hd, h, cd, c, wd, w, yd, y, hod, ho, cod, co,
                         workspace, reserve=nothing) where T
  if reserve == nothing
    @check ccall((:cudnnRNNForwardInference, libcudnn), cudnnStatus_t,
                 (Ptr{Nothing}, Ptr{Nothing}, Cint,
                  Ptr{Ptr{Nothing}}, Ptr{T}, Ptr{Nothing}, Ptr{T}, Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Ptr{T}, Ptr{Ptr{Nothing}}, Ptr{T}, Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Csize_t),
                 handle(), rnn, seqlen,
                 xd, x, hd, h, cd, c, wd, w, yd, y, hod, ho, cod, co,
                 workspace, length(workspace))
  else
    @check ccall((:cudnnRNNForwardTraining, libcudnn), cudnnStatus_t,
                 (Ptr{Nothing}, Ptr{Nothing}, Cint,
                  Ptr{Ptr{Nothing}}, Ptr{T}, Ptr{Nothing}, Ptr{T}, Ptr{Nothing}, Ptr{T}, Ptr{Nothing}, Ptr{T}, Ptr{Ptr{Nothing}}, Ptr{T}, Ptr{Nothing}, Ptr{T}, Ptr{Nothing}, Ptr{T},
                  Ptr{Nothing}, Csize_t, Ptr{Nothing}, Csize_t),
                 handle(), rnn, seqlen,
                 xd, x, hd, h, cd, c, wd, w, yd, y, hod, ho, cod, co,
                 workspace, length(workspace), reserve, length(reserve))
  end
end

xDesc(x) = [TensorDesc(eltype(x), (1, size(x, 1), size(x, 2)))]

hDesc(h::Nothing) = C_NULL, C_NULL
hDesc(x::Integer) = (@assert x == 0; hDesc(nothing))
function hDesc(h::CuArray)
  TensorDesc(eltype(h), (size(h, 1), size(h, 2), 1)), h
end

# TODO: can we just manipulate strides here?
# TODO: should use repmat, but this isn't implemented.
hBatch(x::AbstractVector, h::CuVector) = h
hBatch(x::AbstractMatrix, h::CuVector) = h .* cuones(1, size(x, 2))
hBatch(x::AbstractMatrix, h::CuMatrix) = h .* cuones(1, size(h,2) == 1 ? size(x,2) : 1)

function forward(rnn::RNNDesc{T}, x::CuArray{T}, h_::CuArray{T}, c_ = nothing, train = Val{false}) where T
  h = hBatch(x, h_)
  c = c_ == nothing ? nothing : hBatch(x, c_)
  @assert size(x, 1) == rnn.input
  @assert size(h, 1) == rnn.hidden
  @assert size(x, 2) == size(h, 2)
  seqLength = 1
  xdesc = xDesc(x)
  y = x isa AbstractVector ? similar(x, rnn.hidden) : similar(x, rnn.hidden, size(x, 2))
  ho = similar(h)
  ydesc = xDesc(y)
  workspace = getworkspace(rnn, seqLength, xdesc)
  reserve = train == Val{true} ?
    CuVector{UInt8}(undef, rnnTrainingReserveSize(rnn, seqLength, xdesc)) :
    nothing
  co = c == nothing ? c : similar(c)
  cudnnRNNForward(rnn, seqLength,
                  xdesc, x,
                  hDesc(h)...,
                  hDesc(c)...,
                  FilterDesc(T, (1, 1, length(rnn.params))), rnn.params,
                  ydesc, y,
                  hDesc(ho)...,
                  hDesc(co)...,
                  workspace, reserve)
  result = c == nothing ? (y, ho) : (y, ho, co)
  return train == Val{true} ? (reserve, result) : result
end

forwardTrain(rnn::RNNDesc{T}, x::CuArray{T}, h::CuArray{T}, c = nothing) where T =
  forward(rnn, x, h, c, Val{true})

function cudnnRNNBackwardData(rnn::RNNDesc{T}, seqlen, yd, y, dyd, dy, dhod, dho, dcod, dco,
                              wd, w, hd, h, cd, c, dxd, dx, dhd, dh, dcd, dc, ws, rs) where T
  @check ccall((:cudnnRNNBackwardData,libcudnn),cudnnStatus_t,
               (Ptr{Nothing}, Ptr{Nothing}, Cint,
                Ptr{Ptr{Nothing}}, Ptr{T}, Ptr{Ptr{Nothing}}, Ptr{T}, Ptr{Nothing}, Ptr{T},
                Ptr{Nothing}, Ptr{T}, Ptr{Nothing}, Ptr{T}, Ptr{Nothing}, Ptr{T}, Ptr{Nothing},
                Ptr{T}, Ptr{Ptr{Nothing}}, Ptr{T}, Ptr{Nothing}, Ptr{T}, Ptr{Nothing}, Ptr{T},
                Ptr{Nothing}, Csize_t, Ptr{Nothing}, Csize_t),
               handle(), rnn, seqlen, yd, y, dyd, dy, dhod, dho, dcod, dco,
               wd, w, hd, h, cd, c, dxd, dx, dhd, dh, dcd, dc, ws, length(ws), rs, length(rs))
end

function backwardData(rnn::RNNDesc{T}, y, dy_, dho, dco, h, c, reserve) where T
  # Same as above, any more efficient way?
  dy = dy_ isa Integer ? zero(y) : dy_
  yd = xDesc(y)
  dx = y isa AbstractVector ? similar(dy, rnn.input) : similar(dy, rnn.input, size(dy, 2))
  dh = similar(h)
  dc = c == nothing ? nothing : similar(c)
  cudnnRNNBackwardData(rnn, 1,
    yd, y, yd, dy, hDesc(dho)..., hDesc(dco)...,
    FilterDesc(T, (1, 1, length(rnn.params))), rnn.params,
    hDesc(h)..., hDesc(c)..., xDesc(dx), dx, hDesc(dh)..., hDesc(dc)...,
    workspace[], reserve)
  return c == nothing ? (dx, dh) : (dx, dh, dc)
end

backwardData(rnn, y, dy, dho, hx, reserve) =
  backwardData(rnn, y, dy, dho, nothing, hx, nothing, reserve)

function cudnnRNNBackwardWeights(rnn::RNNDesc{T}, seqlen, xd, x, hd, h, yd, y, dwd, dw,
                                 workspace, reserve) where T
  @check ccall((:cudnnRNNBackwardWeights,libcudnn), cudnnStatus_t,
               (Ptr{Nothing}, Ptr{Nothing}, Cint,  # handle, rnnDesc, seqLength
                Ptr{Ptr{Nothing}}, Ptr{T}, #x
                Ptr{Nothing}, Ptr{T}, #hx
                Ptr{Ptr{Nothing}}, Ptr{T}, #y
                Ptr{Nothing}, Csize_t, #ws
                Ptr{Nothing}, Ptr{T}, #dw
                Ptr{Nothing}, Csize_t), #rs
               handle(), rnn, seqlen, xd, x, hd, h, yd, y,
               workspace, length(workspace), dwd, dw, reserve, length(reserve))
end

function backwardWeights(rnn::RNNDesc{T}, x, h, y, reserve) where T
  dw = zero(rnn.params)
  cudnnRNNBackwardWeights(rnn, 1,
    xDesc(x), x, hDesc(h)..., xDesc(y), y,
    FilterDesc(T, (1, 1, length(dw))), dw,
    workspace[], reserve)
  return params(dw, rnn.input, rnn.hidden, ngates(rnn))
end

# Interface

import ..Flux: Flux, relu
import ..Tracker: TrackedArray
using .CuArrays.CUDAnative
using .CuArrays: @cuindex, cudims

function LinearAlgebra.copy_transpose!(dst::CuArray, src::CuArray)
  function kernel(dst, src)
    I = @cuindex dst
    dst[I...] = src[reverse(I)...]
    return
  end
end

CuParam{T,N} = Union{CuArray{T,N},TrackedArray{T,N,CuArray{T,N}}}
CuRNN{T} = Flux.RNNCell{<:Union{typeof(tanh),typeof(relu)},<:CuParam{T,2},<:CuParam{T,1}}
CuGRU{T} = Flux.GRUCell{<:CuParam{T,2},<:CuParam{T,1}}
CuLSTM{T} = Flux.LSTMCell{<:CuParam{T,2},<:CuParam{T,1}}
CuRNNs{T} = Union{CuRNN{T},CuGRU{T},CuLSTM{T}}

function copyparams!(m::CuRNNs, d::RNNDesc)
  Wi, Wh = d.weights
  copy_transpose!(Wi, Flux.data(m.Wi))
  copy_transpose!(Wh, Flux.data(m.Wh))
  copy_transpose!(d.bias, Flux.data(m.b))
  return
end

function RNNDesc(m::CuRNNs{T}) where T
  h, i = length(m.h), size(m.Wi, 2)
  mode = m isa CuRNN ?
    (m.σ == tanh ? RNN_TANH : RNN_RELU) :
    m isa CuGRU ? GRU : LSTM
  r = RNNDesc{T}(mode, i, h)
  return r
end

const descs = WeakKeyDict()

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

(m::CuRNN{T})(h::CuParam{T}, x) where T <: Union{Float32,Float64} = m(h, CuArray{T}(x))
(m::CuGRU{T})(h::CuParam{T}, x) where T <: Union{Float32,Float64} = m(h, CuArray{T}(x))
(m::CuLSTM{T})(h::NTuple{2,CuParam{T}}, x) where T <: Union{Float32,Float64} = m(h, CuArray{T}(x))

@grad function (m::Union{CuRNN,CuGRU})(x, h, Wi, Wh, b)
  reserve, result = forwardTrain(desc(m), data(x), data(h))
  result, function (Δ)
    y, ho = result
    dy, dho = Δ
    h_ = hBatch(x, data(h))
    dx, dh = backwardData(descs[m], y, dy, dho, h_, reserve)
    (dWi, dWh), db = backwardWeights(descs[m], data(x), h_, y, reserve)
    nobacksies(:RNN, (dx, unbroadcast(h, dh), transpose(dWi), transpose(dWh), db))
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
      (dx, unbroadcast(h, dh), unbroadcast(c, dc),
       transpose(dWi), transpose(dWh), db))
  end
end
