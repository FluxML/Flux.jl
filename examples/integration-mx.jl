using Flux, MXNet

Flux.loadmx()

conv1 = Chain(
  Input(28,28),
  Conv2D((5,5), out = 20), tanh,
  MaxPool((2,2), stride = (2,2)))

conv2 = Chain(
  conv1,
  Conv2D((5,5), in = 20, out = 50), tanh,
  MaxPool((2,2), stride = (2,2)))

lenet = Chain(
  conv2,
  flatten,
  Affine(500), tanh,
  Affine(10), softmax)

#--------------------------------------------------------------------------------

# Now we can continue exactly as in plain MXNet, following
#   https://github.com/dmlc/MXNet.jl/blob/master/examples/mnist/lenet.jl

batch_size = 100
include(Pkg.dir("MXNet", "examples", "mnist", "mnist-data.jl"))
train_provider, eval_provider = get_mnist_providers(batch_size; flat=false)

model = mx.FeedForward(lenet)

mx.infer_shape(model.arch, data = (28, 28, 1, 100))

optimizer = mx.SGD(lr=0.05, momentum=0.9, weight_decay=0.00001)

mx.fit(model, optimizer, train_provider, n_epoch=1, eval_data=eval_provider)
