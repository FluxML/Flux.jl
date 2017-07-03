using Flux, MNIST
using Flux: accuracy, onehot

data = [(trainfeatures(i), onehot(trainlabel(i), 0:9)) for i = 1:60_000]
train = data[1:50_000]
test = data[50_001:60_000]

m = @Chain(
  Input(784),
  Affine(128), relu,
  Affine( 64), relu,
  Affine( 10), softmax)

# Convert to MXNet
model = mxnet(m)

# An example prediction pre-training
model(tobatch(data[1][1]))

Flux.train!(model, train, η = 1e-3,
            cb = [()->@show accuracy(m, test)])

# An example prediction post-training
model(tobatch(data[1][1]))
