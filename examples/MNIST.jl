using Flux, MNIST

data = [(trainfeatures(i), Vector{Float64}(onehot(trainlabel(i), 0:9))) for i = 1:60_000]
train = data[1:50_000]
test = data[50_001:60_000]

m = Chain(
  Input(784),
  Dense(128), relu,
  Dense( 64), relu,
  Dense( 10), softmax)

# Convert to TensorFlow
model = tf(m)

@time Flux.train!(model, train, test, η = 1e-3)
