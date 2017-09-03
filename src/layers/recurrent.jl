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

struct LSTMCell{D1,D2,V}
  forget::D1
  input::D1
  output::D1
  cell::D2
  h::V; c::V
end

LSTMCell(in, out; init = initn) =
  LSTMCell([Dense(in+out, out, σ, init = initn) for _ = 1:3]...,
           Dense(in+out, out, tanh, init = initn),
           track(zeros(out)), track(zeros(out)))

function (m::LSTMCell)(h_, x)
  h, c = h_
  x′ = [x; h]
  forget, input, output, cell =
    m.forget(x′), m.input(x′), m.output(x′), m.cell(x′)
  c = forget .* c .+ input .* cell
  h = output .* tanh.(c)
  return (h, c), h
end

hidden(m::LSTMCell) = (m.h, m.c)
