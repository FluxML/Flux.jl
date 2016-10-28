using Flux

input = readstring("$(homedir())/Downloads/shakespeare_input.txt")

alphabet = unique(input)

getseqs(data, n) = (Seq(onehot(Float32, char, alphabet) for char in chunk) for chunk in chunks(data, n))

data = zip(getseqs(input, 50), getseqs(input[2:end], 50))

model = Chain(
  Input(length(alphabet)),
  Flux.Recurrent(length(alphabet), 128, length(alphabet)),
  softmax)

unrolled = unroll(model, 50)

m = tf(unrolled)

Flux.train!(m, data)
