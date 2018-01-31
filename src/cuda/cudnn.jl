using CuArrays.CUDNN: @check, libcudnn, cudnnStatus_t, libcudnn_handle,
  cudnnDataType, TensorDesc, FilterDesc

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
  slice(offset, shape) = reshape(w[offset+(1:prod(shape))], shape)
  wx = slice(0, (input, hidden*n))
  wh = slice(length(wx), (hidden, hidden*n))
  bias = w[length(wx)+length(wh) + (1:hidden*n)]
  (wx, wh), bias
end

mutable struct RNNDesc{T}
  mode::Int
  input::Int
  hidden::Int
  params::CuVector{T}
  weights::NTuple{2,CuMatrix{T}}
  bias::CuVector{T}
  ptr::Ptr{Void}
end

Base.unsafe_convert(::Type{Ptr{Void}}, d::RNNDesc) = d.ptr

function rnnParamSize(T, r, input)
  size = Csize_t[0]
  @check ccall((:cudnnGetRNNParamsSize, libcudnn), cudnnStatus_t, (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Csize_t},Cint),
    libcudnn_handle[], r, TensorDesc(T, (1,input,1)), size, cudnnDataType(T))
  return Int(size[])÷sizeof(T)
end

function RNNDesc{T}(mode::Int, input::Int, hidden::Int; layers = 1) where T
  d = [C_NULL]
  @check ccall((:cudnnCreateRNNDescriptor,libcudnn),cudnnStatus_t,(Ptr{Ptr{Void}},),d)

  dropoutDesc = DropoutDesc(0)
  inputMode = LINEAR_INPUT
  direction = UNIDIRECTIONAL
  algo = RNN_ALGO_STANDARD
  @check ccall((:cudnnSetRNNDescriptor_v6,libcudnn), cudnnStatus_t, (Ptr{Void},Ptr{Void},Cint,Cint,Ptr{Void},Cint,Cint,Cint,Cint,Cint),
    libcudnn_handle[],d[],hidden,layers,dropoutDesc,inputMode,direction,mode,algo,cudnnDataType(T))

  w = cuzeros(T, rnnParamSize(T, d[], 10))
  ngates = [1, 1, 4, 3][mode+1]
  rd = RNNDesc{T}(mode, input, hidden, w, params(w, input, hidden, ngates)..., d[])
  finalizer(rd, x ->
    @check ccall((:cudnnDestroyRNNDescriptor,libcudnn),cudnnStatus_t,(Ptr{Void},),x))
  return rd
end

function rnnWorkspaceSize(r::RNNDesc, seqlen, xdesc)
  size = Csize_t[0]
  @check ccall((:cudnnGetRNNWorkspaceSize, libcudnn), cudnnStatus_t, (Ptr{Void},Ptr{Void},Cint,Ptr{Ptr{Void}},Ptr{Csize_t}),
    libcudnn_handle[], r, seqlen, xdesc, size)
  return Int(size[])
end

function rnnTrainingReserveSize(r::RNNDesc, seqlen, xdesc)
  size = Csize_t[0]
  @check ccall((:cudnnGetRNNTrainingReserveSize,libcudnn), cudnnStatus_t, (Ptr{Void}, Ptr{Void}, Cint, Ptr{Ptr{Void}}, Ptr{Csize_t}),
    libcudnn_handle[], r, seqlen, xdesc, size)
  return Int(size[])
end

function forwardInference(rnn::RNNDesc{T}, x, h, c = nothing) where T
  @assert size(x, 1) == rnn.input
  @assert size(h, 1) == rnn.hidden
  @assert size(x, 2) == size(h, 2)
  seqLength = 1
  xdesc = [TensorDesc(reshape(x, 1, size(x, 1), size(x, 2)))]
  y = x isa AbstractVector ? similar(x, rnn.hidden) : similar(x, rnn.hidden, size(x, 2))
  ydesc = [TensorDesc(reshape(y, 1, size(y, 1), size(y, 2)))]
  hout = similar(h)
  workspace = CuVector{UInt8}(rnnWorkspaceSize(rnn, seqLength, xdesc)) # TODO: reuse this
  if c ≠ nothing
    @assert size(c, 1) == rnn.hidden
    @assert size(c, 2) == size(h, 2)
    cptr = c
    cdesc = TensorDesc(reshape(c, size(c, 1), size(c, 2), 1))
    cout = similar(c)
    coutdesc = TensorDesc(reshape(cout, size(cout, 1), size(cout, 2), 1))
  else
    cptr = cdesc = cout = coutdesc = C_NULL
  end
  @check ccall((:cudnnRNNForwardInference, libcudnn), cudnnStatus_t,
               (Ptr{Void}, Ptr{Void}, Cint,
                Ptr{Ptr{Void}}, Ptr{T},
                Ptr{Void}, Ptr{T},
                Ptr{Void}, Ptr{T},
                Ptr{Void}, Ptr{T},
                Ptr{Ptr{Void}}, Ptr{T},
                Ptr{Void}, Ptr{T},
                Ptr{Void}, Ptr{T},
                Ptr{Void}, Csize_t),
               libcudnn_handle[], rnn, seqLength,
               xdesc, x,
               TensorDesc(reshape(h, size(h, 1), size(h, 2), 1)), h,
               cdesc, cptr,
               TensorDesc(reshape(rnn.params, 1, 1, :)), rnn.params,
               ydesc, y,
               TensorDesc(reshape(hout, size(hout, 1), size(hout, 2), 1)), hout,
               coutdesc, cout,
               workspace, length(workspace))
  if c == nothing
    return y, hout
  else
    return y, hout, cout
  end
end
