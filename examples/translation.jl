# Based on https://arxiv.org/abs/1409.0473

using Flux
using Flux: flip, stateless, broadcastto, ∘

Nbatch  =  3 # Number of phrases to batch together
Nphrase =  5 # The length of (padded) phrases
Nalpha  =  7 # The size of the token vector
Nhidden = 10 # The size of the hidden state

# A recurrent model which takes a token and returns a context-dependent
# annotation.

forward  = LSTM(Nalpha, Nhidden÷2)
backward = flip(LSTM(Nalpha, Nhidden÷2))
encoder  = @net token -> hcat(forward(token), backward(token))

alignnet = Affine(2Nhidden, 1)
align  = @net (s, t) -> alignnet(hcat(broadcastto(s, (Nbatch, 1)), t))

# A recurrent model which takes a sequence of annotations, attends, and returns
# a predicted output token.

recur   = unroll1(LSTM(Nhidden, Nhidden)).model
state   = param(zeros(1, Nhidden))
y       = param(zeros(1, Nhidden))
toalpha = Affine(Nhidden, Nalpha)

decoder = @net function (tokens)
  energies = map(token -> exp.(align(state{-1}, token)), tokens)
  weights = map(e -> e ./ sum(energies), energies)
  context = sum(map(∘, weights, tokens))
  (y, state), _ = recur((y{-1},state{-1}), context)
  return softmax(toalpha(y))
end

# Building the full model

a, b = rand(Nbatch, Nalpha), rand(Nbatch, Nalpha)

model = @Chain(
  stateless(unroll(encoder, Nphrase)),
  @net(x -> repeated(x, Nphrase)),
  stateless(unroll(decoder, Nphrase)))

model = mxnet(Flux.SeqModel(model, Nphrase))

xs = Batch(Seq(rand(Nalpha) for i = 1:Nphrase) for i = 1:Nbatch)

model(xs)
