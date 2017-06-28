using Flux
using Flux: onehot, logloss, unsqueeze
using Flux.Batches: Batch, tobatch, seqs, chunk
import StatsBase: wsample

nunroll = 50
nbatch = 50

encode(input) = seqs((onehot(ch, alphabet) for ch in input), nunroll)

cd(@__DIR__)
input = readstring("shakespeare_input.txt");
alphabet = unique(input)
N = length(alphabet)

Xs = (Batch(ss) for ss in zip(encode.(chunk(input, 50))...))
Ys = (Batch(ss) for ss in zip(encode.(chunk(input[2:end], 50))...))

model = Chain(
  LSTM(N, 256),
  LSTM(256, 256),
  Affine(256, N),
  softmax)

m = mxnet(unroll(model, nunroll))

eval = tobatch.(first.(drop.((Xs, Ys), 5)))
evalcb = () -> @show logloss(m(eval[1]), eval[2])

# @time Flux.train!(m, zip(Xs, Ys), η = 0.001, loss = logloss, cb = [evalcb], epoch = 10)

function sample(model, n, temp = 1)
  s = [rand(alphabet)]
  m = unroll1(model)
  for i = 1:n-1
    push!(s, wsample(alphabet, softmax(m(unsqueeze(onehot(s[end], alphabet)))./temp)[1,:]))
  end
  return string(s...)
end

# s = sample(model[1:end-1], 100)
