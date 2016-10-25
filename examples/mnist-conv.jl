using Flux, Juno

# Flux aims to provide high-level APIs that work well across backends, but in
# some cases you may want to take advantage of features specific to a given
# backend (or Flux may simply not have an implementation of that feature yet).
# In these cases it's easy to "drop down" and use the backend's API directly,
# where appropriate.

# In this example, both things are happening; firstly, Flux doesn't yet support
# ConvNets in the pure-Julia backend, but this is invisible thanks to the use of
# a simple "shim" type, `Conv2D`. This is provided by the library but could easily
# have been user-defined.

# Secondly, we want to take advantage of TensorFlow.jl's training process and
# optimisers. We can simply call `Tensor` exactly as we would on a regular
# TensorFlow model, and the rest of the process trivially follows
# TensorFlow.jl's usual API.

conv1 = Chain(
  Reshape(28,28,1),
  Conv2D((5,5), out = 20), tanh,
  MaxPool((2,2), stride = (2,2)))

conv2 = Chain(
  Input(12,12,20),
  Conv2D((5,5), in = 20, out = 50), tanh,
  MaxPool((2,2), stride = (2,2)))

lenet = Chain(
  conv1, conv2, flatten,
  Dense(500), tanh,
  Dense(10), softmax)

#--------------------------------------------------------------------------------

# Now we can continue exactly as in plain TensorFlow, following
#   https://github.com/malmaud/TensorFlow.jl/blob/master/examples/mnist_full.jl
# (taking only the training and cost logic, not the graph building steps)

using TensorFlow, Distributions

include(Pkg.dir("TensorFlow", "examples", "mnist_loader.jl"))
loader = DataLoader()

session = Session(Graph())

x  = placeholder(Float32)
y′ = placeholder(Float32)
y  = Tensor(lenet, x)

cross_entropy = reduce_mean(-reduce_sum(y′.*log(y), reduction_indices=[2]))

train_step = train.minimize(train.AdamOptimizer(1e-4), cross_entropy)

accuracy = reduce_mean(cast(indmax(y, 2) .== indmax(y′, 2), Float32))

run(session, initialize_all_variables())

@progress for i in 1:1000
    batch = next_batch(loader, 50)
    if i%100 == 1
        train_accuracy = run(session, accuracy, Dict(x=>batch[1], y′=>batch[2]))
        info("step $i, training accuracy $train_accuracy")
    end
    run(session, train_step, Dict(x=>batch[1], y′=>batch[2]))
end

testx, testy = load_test_set()
test_accuracy = run(session, accuracy, Dict(x=>testx, y′=>testy))
info("test accuracy $test_accuracy")
