using Flux
import StatsBase: wsample

nunroll = 50
nbatch = 50

getseqs(chars, alphabet) =
  sequences((onehot(Float32, char, alphabet) for char in chars), nunroll)
getbatches(chars, alphabet) =
  batches((getseqs(part, alphabet) for part in chunk(chars, nbatch))...)

input = readstring("$(homedir())/Downloads/shakespeare_input.txt");
alphabet = unique(input)
N = length(alphabet)

train = zip(getbatches(input, alphabet), getbatches(input[2:end], alphabet))

model = Chain(
  Input(N),
  LSTM(N, 256),
  LSTM(256, 256),
  Affine(256, N),
  softmax)

m = mxnet(unroll(model, nunroll))

@time Flux.train!(m, train, η = 0.1, loss = logloss)

function sample(model, n, temp = 1)
  s = [rand(alphabet)]
  m = unroll1(model)
  for i = 1:n-1
    push!(s, wsample(alphabet, softmax(m(unsqueeze(onehot(s[end], alphabet)))./temp)[1,:]))
  end
  return string(s...)
end

s = sample(model[1:end-1], 100)
