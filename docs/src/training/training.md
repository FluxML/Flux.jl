# Training

To actually train a model we need three things:

* A *model loss function*, that evaluates how well a model is doing given some input data.
* A collection of data points that will be provided to the loss function.
* An [optimiser](optimisers.md) that will update the model parameters appropriately.

With these we can call `Flux.train!`:

```julia
Flux.train!(modelLoss, data, opt)
```

There are plenty of examples in the [model zoo](https://github.com/FluxML/model-zoo).

## Loss Functions

The `loss` that we defined in [basics](../models/basics.md) is completely valid for training. We can also define a loss in terms of some model:

```julia
m = Chain(
  Dense(784, 32, σ),
  Dense(32, 10), softmax)

# Model loss function
loss(x, y) = Flux.mse(m(x), y)

# later
Flux.train!(loss, data, opt)
```

The loss will almost always be defined in terms of some *cost function* that measures the distance of the prediction `m(x)` from the target `y`. Flux has several of these built in, like `mse` for mean squared error or `crossentropy` for cross entropy loss, but you can calculate it however you want.

## Datasets

The `data` argument provides a collection of data to train with (usually a set of inputs `x` and target outputs `y`). For example, here's a dummy data set with only one data point:

```julia
x = rand(784)
y = rand(10)
data = [(x, y)]
```

`Flux.train!` will call `loss(x, y)`, calculate gradients, update the weights and then move on to the next data point if there is one. We can train the model on the same data three times:

```julia
data = [(x, y), (x, y), (x, y)]
# Or equivalently
data = Iterators.repeated((x, y), 3)
```

It's common to load the `x`s and `y`s separately. In this case you can use `zip`:

```julia
xs = [rand(784), rand(784), rand(784)]
ys = [rand( 10), rand( 10), rand( 10)]
data = zip(xs, ys)
```

## Callbacks

`train!` takes an additional argument, `cb`, that's used for callbacks so that you can observe the training process. For example:

```julia
train!(loss, data, opt, cb = () -> println("training"))
```

Callbacks are called for every batch of training data. You can slow this down using `Flux.throttle(f, timeout)` which prevents `f` from being called more than once every `timeout` seconds.

A more typical callback might look like this:

```julia
test_x, test_y = # ... create single batch of test data ...
evalcb() = @show(loss(test_x, test_y))

Flux.train!(loss, data, opt,
            cb = throttle(evalcb, 5))
```
