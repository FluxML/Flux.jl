using Flux, MNIST

const data = collect(zip([trainfeatures(i) for i = 1:60_000],
                         [onehot(trainlabel(i), 1:10) for i = 1:60_000]))
const train = data[1:50_000]
const test = data[50_001:60_000]

const m = Sequence(
  Input(784),
  Dense(30), Sigmoid(),
  Dense(10), Sigmoid())

@time Flux.train!(m, train, test, epoch = 30)
