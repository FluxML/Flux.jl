# Stateful recurrence

mutable struct Recur{T}
  cell::T
  state
end

Recur(m) = Recur(m, hidden(m))

function (m::Recur)(xs...)
  h, y = m.cell(m.state, xs...)
  m.state = h
  return y
end

# Vanilla RNN

struct RNNCell{D,V}
  d::D
  h::V
end

RNNCell(in::Integer, out::Integer, init = initn) =
  RNNCell(Dense(in+out, out, init = initn), track(initn(out)))

function (m::RNNCell)(h, x)
  h = m.d([x; h])
  return h, h
end

hidden(m::RNNCell) = m.h

function Base.show(io::IO, m::RNNCell)
  print(io, "RNNCell(", m.d, ")")
end

# LSTM

struct LSTMCell{M}
  Wxf::M; Wyf::M; bf::M
  Wxi::M; Wyi::M; bi::M
  Wxo::M; Wyo::M; bo::M
  Wxc::M; Wyc::M; bc::M
  hidden::M; cell::M
end

LSTMCell(in, out; init = initn) =
  LSTMCell(track.(vcat([[init(out, in), init(out, out), init(out, 1)] for _ = 1:4]...))...,
       track(zeros(out, 1)), track(zeros(out, 1)))

function (m::LSTMCell)(h_, x)
  h, c = h_
  # Gates
  forget = σ.( m.Wxf * x .+ m.Wyf * h .+ m.bf )
  input  = σ.( m.Wxi * x .+ m.Wyi * h .+ m.bi )
  output = σ.( m.Wxo * x .+ m.Wyo * h .+ m.bo )
  # State update and output
  c′ = tanh.( m.Wxc * x .+ m.Wyc * h .+ m.bc )
  c  = forget .* c .+ input .* c′
  h = output .* tanh.(c)
  return (h, c), h
end

hidden(m::LSTMCell) = (m.hidden, m.cell)
