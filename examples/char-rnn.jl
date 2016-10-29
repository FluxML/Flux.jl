using Flux

using Juno

getseqs(chars, alphabet) = sequences((onehot(Float32, char, alphabet) for char in chars), 50)
getbatches(chars, alphabet) = batches((getseqs(part, alphabet) for part in chunk(chars, 50))...)

input = readstring("$(homedir())/Downloads/shakespeare_input.txt")
const alphabet = unique(input)

train = zip(getbatches(input, alphabet), getbatches(input[2:end], alphabet))

model = Chain(
  Input(length(alphabet)),
  Flux.Recurrent(length(alphabet), 128, length(alphabet)),
  softmax)

m = tf(unroll(model, 50))

Flux.train!(m, train, η = 0.1/50, epoch = 5)

map(c->onecold(c, alphabet), m(train[1][1][1]))
