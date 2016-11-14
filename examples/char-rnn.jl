using Flux
import StatsBase: wsample

getseqs(chars, alphabet) = sequences((onehot(Float32, char, alphabet) for char in chars), 50)
getbatches(chars, alphabet) = batches((getseqs(part, alphabet) for part in chunk(chars, 50))...)

input = readstring("$(homedir())/Downloads/shakespeare_input.txt")
alphabet = unique(input)
N = length(alphabet)

Xs, Ys = getbatches(input, alphabet), getbatches(input[2:end], alphabet)

model = Chain(
  Input(N),
  LSTM(N, 256),
  LSTM(256, 256),
  Affine(256, N),
  softmax)

m = tf(unroll(model, 50));

@time Flux.train!(m, Xs, Ys, η = 0.1, epoch = 1)

string(map(c -> onecold(c, alphabet), m(first(first(Xs))))...)

function sample(model, n)
  s = [rand(alphabet)]
  m = tf(unroll(model, 1))
  for i = 1:n
    push!(s, wsample(alphabet, m(Seq((onehot(Float32, s[end], alphabet),)))[1]))
  end
  return string(s...)
end

sample(model, 100)
