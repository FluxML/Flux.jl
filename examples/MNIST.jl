using Flux, MNIST

data = [(Vector{Float32}(trainfeatures(i)), onehot(Float32, trainlabel(i), 0:9)) for i = 1:60_000]
train = data[1:50_000]
test = data[50_001:60_000]

m = Chain(
  Input(784),
  Affine(128), relu,
  Affine( 64), relu,
  Affine( 10), softmax)

# Convert to TensorFlow
model = tf(m)

# An example prediction pre-training
model(data[1][1])

@time Flux.train!(model, train, test, η = 1e-3)

# An example prediction post-training
model(data[1][1])
