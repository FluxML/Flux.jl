# Char RNN

```julia
using Flux
import StatsBase: wsample

nunroll = 50
nbatch = 50

getseqs(chars, alphabet) = sequences((onehot(Float32, char, alphabet) for char in chars), nunroll)
getbatches(chars, alphabet) = batches((getseqs(part, alphabet) for part in chunk(chars, nbatch))...)

input = readstring("$(homedir())/Downloads/shakespeare_input.txt")
alphabet = unique(input)
N = length(alphabet)

Xs, Ys = getbatches(input, alphabet), getbatches(input[2:end], alphabet)

basemodel = Chain(
  Input(N),
  LSTM(N, 256),
  LSTM(256, 256),
  Affine(256, N),
  softmax)

model = Chain(basemodel, softmax)

m = tf(unroll(model, nunroll))

@time Flux.train!(m, Xs, Ys, η = 0.1, epoch = 1)

function sample(model, n, temp = 1)
  s = [rand(alphabet)]
  m = tf(unroll(model, 1))
  for i = 1:n
    push!(s, wsample(alphabet, softmax(m(Seq((onehot(Float32, s[end], alphabet),)))[1]./temp)))
  end
  return string(s...)
end

sample(basemodel, 100)
```
