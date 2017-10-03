# TODO: broadcasting cat
combine(x, h) = vcat(x, h .* trues(1, size(x, 2)))

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

treelike(Recur)

Base.show(io::IO, m::Recur) = print(io, "Recur(", m.cell, ")")

_truncate(x::AbstractArray) = x
_truncate(x::TrackedArray) = x.data
_truncate(x::Tuple) = _truncate.(x)

truncate!(m) = foreach(truncate!, children(m))
truncate!(m::Recur) = (m.state = _truncate(m.state))

# Vanilla RNN

struct RNNCell{D,V}
  d::D
  h::V
end

RNNCell(in::Integer, out::Integer, σ = tanh; init = initn) =
  RNNCell(Dense(in+out, out, σ, init = init), param(init(out)))

function (m::RNNCell)(h, x)
  h = m.d(combine(x, h))
  return h, h
end

hidden(m::RNNCell) = m.h

treelike(RNNCell)

function Base.show(io::IO, m::RNNCell)
  print(io, "RNNCell(", m.d, ")")
end

RNN(a...; ka...) = Recur(RNNCell(a...; ka...))

# LSTM

struct LSTMCell{D1,D2,V}
  forget::D1
  input::D1
  output::D1
  cell::D2
  h::V; c::V
end

function LSTMCell(in, out; init = initn)
  cell = LSTMCell([Dense(in+out, out, σ, init = init) for _ = 1:3]...,
                  Dense(in+out, out, tanh, init = init),
                  param(init(out)), param(init(out)))
  cell.forget.b.data .= 1
  return cell
end

function (m::LSTMCell)(h_, x)
  h, c = h_
  x′ = combine(x, h)
  forget, input, output, cell =
    m.forget(x′), m.input(x′), m.output(x′), m.cell(x′)
  c = forget .* c .+ input .* cell
  h = output .* tanh.(c)
  return (h, c), h
end

hidden(m::LSTMCell) = (m.h, m.c)

treelike(LSTMCell)

Base.show(io::IO, m::LSTMCell) =
  print(io, "LSTMCell(",
        size(m.forget.W, 2) - size(m.forget.W, 1), ", ",
        size(m.forget.W, 1), ')')

LSTM(a...; ka...) = Recur(LSTMCell(a...; ka...))
