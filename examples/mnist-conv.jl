using Flux, MXNet

# Flux aims to provide high-level APIs that work well across backends, but in
# some cases you may want to take advantage of features specific to a given
# backend (or alternatively, Flux may simply not have an implementation of that
# feature yet). In these cases it's easy to "drop down" and use the backend's
# API directly, where appropriate.

# In this example, both things are happening; firstly, Flux doesn't yet support
# ConvNets in the pure-Julia backend, but this is invisible thanks to the use of
# a simple "shim" type, `Conv`. This is provided by the library but could easily
# have been user-defined.

# Secondly, we want to take advantage of MXNet.jl's training process and
# optimisers. We can simply call `mx.FeedForward` exactly as we would on a
# regular MXNet model, and the rest of the process is trivial.

conv1 = Chain(
  Input(28,28),
  Conv((5,5),20), tanh,
  MaxPool((2,2), stride = (2,2)))

conv2 = Chain(
  conv1,
  Conv((5,5),50), tanh,
  MaxPool((2,2), stride = (2,2)))

lenet = Chain(
  conv2,
  flatten,
  Dense(500), tanh,
  Dense(10), softmax)

#--------------------------------------------------------------------------------

# Now we can continue exactly as in plain MXNet, following
#   https://github.com/dmlc/MXNet.jl/blob/master/examples/mnist/lenet.jl

batch_size = 100
include(Pkg.dir("MXNet", "examples", "mnist", "mnist-data.jl"))
train_provider, eval_provider = get_mnist_providers(batch_size; flat=false)

model = mx.FeedForward(lenet, context = mx.gpu())

optimizer = mx.SGD(lr=0.05, momentum=0.9, weight_decay=0.00001)

mx.fit(model, optimizer, train_provider, n_epoch=1, eval_data=eval_provider)
