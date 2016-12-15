# Based on https://arxiv.org/abs/1409.0473

using Flux
using Flux: flip

# A recurrent model which takes a token and returns a context-depedent
# annotation.

@net type Encoder
  forward
  backward
  token -> hcat(forward(token), backward(token))
end

Encoder(in::Integer, out::Integer) =
  Encoder(LSTM(in, out÷2), flip(LSTM(in, out÷2)))

# A recurrent model which takes a sequence of annotations, attends, and returns
# a predicted output token.

@net type Decoder
  attend
  recur
  state; y; N
  function (anns)
    energies = map(ann -> exp(attend(hcat(state{-1}, ann))[1]), seq(anns, N))
    weights = energies./sum(energies)
    ctx = sum(map((α, ann) -> α .* ann, weights, anns))
    (_, state), y = recur((state{-1},y{-1}), ctx)
    y
  end
end

Decoder(in::Integer, out::Integer; N = 1) =
  Decoder(Affine(in+out, 1),
          unroll1(LSTM(in, out)),
          param(zeros(1, out)), param(zeros(1, out)), N)

# The model

Nalpha  =  5 # The size of the input token vector
Nphrase =  7 # The length of (padded) phrases
Nhidden = 12 # The size of the hidden state

encode = Encoder(Nalpha, Nhidden)
decode = Chain(Decoder(Nhidden, Nhidden, N = Nphrase), Affine(Nhidden, Nalpha), softmax)

model = Chain(
  unroll(encode, Nphrase, stateful = false),
  unroll(decode, Nphrase, stateful = false, seq = false))

xs = Batch([Seq(rand(Float32, Nalpha) for _ = 1:Nphrase)])
