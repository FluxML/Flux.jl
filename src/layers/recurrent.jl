# RNN

struct RNN{S,T}
  W::S
  b::T
end

RNN(in::Integer, out::Integer; init = initn) =
  RNN(track(init(in+out, out)), track(zeros(1, out)))

Optimise.children(m::RNN) = (m.W, m.b)

(m::RNN)(x, h) = σ.([x h] * m.W .+ m.b)

Base.show(io::IO, m::RNN) =
  print(io, "RNN(", size(l.W, 1) - size(l.W, 2), ", ", size(l.W, 2), ", ", l.σ, ')')

init_state(m::RNN, batchsize::Integer, init=zeros) =
  init(batchsize, size(m.W, 2))

# GRU

struct GRU{TWi, TWh, Tbi, Tbh}
  Wi::TWi
  Wh::TWh
  bi::Tbi
  bh::Tbh
end

GRU(in::Integer, out::Integer, σ = tanh; init = initn) =
  GRU(track.(init(in, 3out), init(out, 3out), zeros(1, 3out), zeros(1, 3out))...)

Optimise.children(m::GRU) = (m.Wi, m.Wh, m.bi, m.bh)

function (m::GRU)(x, h)
  gi = x * m.Wi .+ m.bi
  gh = h * m.Wh .+ m.bh

  hsize = size(h, 2)

  ir, ii, in = gi[:, 1:hsize], gi[:, 1+hsize:2hsize], gi[:, 1+2hsize:3hsize]
  hr, hi, hn = gh[:, 1:hsize], gh[:, 1+hsize:2hsize], gh[:, 1+2hsize:3hsize]

  rgate = σ.(ir .+ hr)
  igate = σ.(ii .+ hi)
  ngate = tanh.(in .+ rgate .* hn)
  ngate .+ igate .* (h .- ngate)
end

Base.show(io::IO, m::GRU) =
  print(io, "GRU(", size(m.Wi, 1), ", ", size(m.Wh, 1), ')')

init_state(m::GRU, batchsize::Integer, init=zeros) =
  init(batchsize, size(m.Wh, 1))

# LSTM

struct LSTM{S,T}
  W::S
  b::T
end

LSTM(in::Integer, out::Integer; init = initn) =
  LSTM(track(init(in+out, 4out)), track(zeros(1, 4out)))

Optimise.children(m::LSTM) = (m.W, m.b)

function (m::LSTM)(x, h)
  hidden, cell = h

  gates   = [x hidden] * m.W .+ m.b
  hsize   = size(hidden,2)
  forget  = σ.(gates[:,1:hsize])
  ingate  = σ.(gates[:,1+hsize:2hsize])
  outgate = σ.(gates[:,1+2hsize:3hsize])
  change  = tanh.(gates[:,1+3hsize:end])
  cell    = cell .* forget .+ ingate .* change
  hidden  = outgate .* tanh.(cell)
  hidden, cell
end

Base.show(io::IO, m::LSTM) =
  print(io, "LSTM(", size(m.W, 1) - size(m.W, 2) ÷ 4, ", ", size(m.W, 2) ÷ 4, ')')

init_state(m::LSTM, batchsize::Integer, init=zeros) =
  ntuple(i->init(batchsize, size(m.W, 2) ÷ 4), 2)
