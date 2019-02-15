gate(h, n) = (1:h) .+ h*(n-1)
gate(x::AbstractVector, h, n) = x[gate(h,n)]
gate(x::AbstractMatrix, h, n) = x[gate(h,n),:]

# Stateful recurrence

"""
    Recur(cell)

`Recur` takes a recurrent cell and makes it stateful, managing the hidden state
in the background. `cell` should be a model of the form:

    h, y = cell(h, x...)

For example, here's a recurrent network that keeps a running total of its inputs.

```julia
accum(h, x) = (h+x, x)
rnn = Flux.Recur(accum, 0)
rnn(2) # 2
rnn(3) # 3
rnn.state # 5
rnn.(1:10) # apply to a sequence
rnn.state # 60
```
"""
mutable struct Recur{T}
  cell::T
  init
  state
end

Recur(m, h = hidden(m)) = Recur(m, h, h)

function (m::Recur)(xs...)
  h, y = m.cell(m.state, xs...)
  m.state = h
  return y
end

@treelike Recur cell, init

Base.show(io::IO, m::Recur) = print(io, "Recur(", m.cell, ")")

_truncate(x::AbstractArray) = Tracker.data(x)
_truncate(x::Tuple) = _truncate.(x)

"""
    truncate!(rnn)

Truncates the gradient of the hidden state in recurrent layers. The value of the
state is preserved. See also `reset!`.

Assuming you have a `Recur` layer `rnn`, this is roughly equivalent to

    rnn.state = Tracker.data(rnn.state)
"""
truncate!(m) = prefor(x -> x isa Recur && (x.state = _truncate(x.state)), m)

"""
    reset!(rnn)

Reset the hidden state of a recurrent layer back to its original value. See also
`truncate!`.

Assuming you have a `Recur` layer `rnn`, this is roughly equivalent to

    rnn.state = hidden(rnn.cell)
"""
reset!(m) = prefor(x -> x isa Recur && (x.state = x.init), m)

flip(f, xs) = reverse(f.(reverse(xs)))

# Vanilla RNN

mutable struct RNNCell{F,A,V}
  σ::F
  Wi::A
  Wh::A
  b::V
  h::V
end

RNNCell(in::Integer, out::Integer, σ = tanh;
        init = glorot_uniform) =
  RNNCell(σ, param(init(out, in)), param(init(out, out)),
          param(init(out)), param(zeros(out)))

function (m::RNNCell)(h, x)
  σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
  h = σ.(Wi*x .+ Wh*h .+ b)
  return h, h
end

hidden(m::RNNCell) = m.h

@treelike RNNCell

function Base.show(io::IO, l::RNNCell)
  print(io, "RNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    RNN(in::Integer, out::Integer, σ = tanh)

The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""
RNN(a...; ka...) = Recur(RNNCell(a...; ka...))

# LSTM

mutable struct LSTMCell{A,V}
  Wi::A
  Wh::A
  b::V
  h::V
  c::V
end

function LSTMCell(in::Integer, out::Integer;
                  init = glorot_uniform)
  cell = LSTMCell(param(init(out*4, in)), param(init(out*4, out)), param(init(out*4)),
                  param(zeros(out)), param(zeros(out)))
  cell.b.data[gate(out, 2)] .= 1
  return cell
end

function (m::LSTMCell)((h, c), x)
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  return (h′, c), h′
end

hidden(m::LSTMCell) = (m.h, m.c)

@treelike LSTMCell

Base.show(io::IO, l::LSTMCell) =
  print(io, "LSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")

"""
    LSTM(in::Integer, out::Integer)

Long Short Term Memory recurrent layer. Behaves like an RNN but generally
exhibits a longer memory span over sequences.

See [this article](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
LSTM(a...; ka...) = Recur(LSTMCell(a...; ka...))

# GRU

mutable struct GRUCell{A,V}
  Wi::A
  Wh::A
  b::V
  h::V
end

GRUCell(in, out; init = glorot_uniform) =
  GRUCell(param(init(out*3, in)), param(init(out*3, out)),
          param(init(out*3)), param(zeros(out)))

function (m::GRUCell)(h, x)
  b, o = m.b, size(h, 1)
  gx, gh = m.Wi*x, m.Wh*h
  r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
  z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
  h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
  h′ = (1 .- z).*h̃ .+ z.*h
  return h′, h′
end

hidden(m::GRUCell) = m.h

@treelike GRUCell

Base.show(io::IO, l::GRUCell) =
  print(io, "GRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    GRU(in::Integer, out::Integer)

Gated Recurrent Unit layer. Behaves like an RNN but generally
exhibits a longer memory span over sequences.

See [this article](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
GRU(a...; ka...) = Recur(GRUCell(a...; ka...))

# peephole LSTM

mutable struct PeepholeLSTMCell{A,V}
  Wx::A
  Wh::A
  Wc::A
  b::V
  h::V
  c::V
end

function PeepholeLSTMCell(in::Integer, out::Integer;
                          init = glorot_uniform)
  cell = PeepholeLSTMCell(
      param(init(out*4, in)),  # Wx
      param(init(out*4, out)), # Wh
      param(init(out*4, out)), # Wc
      param(zeros(out*4)),     # b
      param(init(out)),        # h
      param(init(out)))        # c
  cell.b.data[gate(out, 2)] .= 1
  return cell
end

function (m::PeepholeLSTMCell)((h, c), x)
  o = size(h, 1)
  g = m.Wx*x .+ m.Wh*h .+ m.b
  g_if = m.Wc*c
  input = σ.(gate(g, o, 1) .+ gate(g_if, o, 1))
  forget = σ.(gate(g, o, 2) .+ gate(g_if, o, 2))
  cell = forget .* c .+ input .* tanh.(gate(g, o, 3))
  g_c = m.Wc*cell
  output = σ.(gate(g, o, 4) .+ gate(g_c, o, 4))
  hidden = output .* tanh.(cell)
  return (hidden, cell), hidden
end

hidden(m::PeepholeLSTMCell) = (m.h, m.c)

@treelike PeepholeLSTMCell

Base.show(io::IO, l::PeepholeLSTMCell) =
  print(io, "PeepholeLSTMCell(", size(l.Wx, 2), ", ", size(l.Wx, 1) ÷ 4, ")")

"""
  PLSTM(in::Integer, out::Integer)

  See the following articles for internals:
  
  * 2000 Felix A. Gers and Jürgen Schmidhuber. „Recurrent nets that time and count“. In Neural Networks, 2000. IJCNN 2000, Proceedings of the IEEE-INNS-ENNS International Joint  Conference  on,  volume  3,  pages  189–194.  IEEE, 2000.  ISBN 0769506194
  * 2005 A. Graves, S. Fernández, and J. Schmidhuber, „Bidirectional LSTM Networks for Improved Phoneme Classification and Recognition“, in Artificial Neural Networks: Formal Models and Their Applications – ICANN 2005, 2005, S. 799–804.
  * 2013 A. Graves, „Generating Sequences With Recurrent Neural Networks“, arXiv:1308.0850 [cs], Aug. 2013.
  * 2015 J. Wieting, M. Bansal, K. Gimpel, and K. Livescu, „Towards Universal Paraphrastic Sentence Embeddings“, arXiv:1511.08198 [cs], Nov. 2015.
"""
PLSTM(a...; ka...) = Recur(PeepholeLSTMCell(a...; ka...))
