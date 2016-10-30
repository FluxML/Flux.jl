using Flux

getseqs(chars, alphabet) = sequences((onehot(Float32, char, alphabet) for char in chars), 50)
getbatches(chars, alphabet) = batches((getseqs(part, alphabet) for part in chunk(chars, 50))...)

input = readstring("$(homedir())/Downloads/shakespeare_input.txt")
alphabet = unique(input)
N = length(alphabet)

Xs, Ys = getbatches(input, alphabet), getbatches(input[2:end], alphabet)

model = Chain(
  Input(N),
  Recurrent(N, 128, N),
  softmax)

m = tf(unroll(model, 50))

Flux.train!(m, Xs, Ys, η = 0.2e-3, epoch = 1)

map(c->onecold(c, alphabet), m(first(first(first(train)))))
