# Training

To actually train a model we need four things:

* A *objective function*, that evaluates how well a model is doing given some input data.
* The trainable parameters of the model.
* A collection of data points that will be provided to the objective function.
* An [optimiser](optimisers.md) that will update the model parameters appropriately.

With these we can call `train!`:

```@docs
Flux.Optimise.train!
```

There are plenty of examples in the [model zoo](https://github.com/FluxML/model-zoo).

## Loss Functions

The objective function must return a number representing how far the model is from its target – the *loss* of the model. The `loss` function that we defined in [basics](../models/basics.md) will work as an objective.
In addition to custom losses, model can be trained in conjuction with
the commonly used losses that are grouped under the `Flux.Losses` module.
We can also define an objective in terms of some model:

```julia
m = Chain(
  Dense(784, 32, σ),
  Dense(32, 10), softmax)

loss(x, y) = Flux.Losses.mse(m(x), y)
ps = Flux.params(m)

# later
Flux.train!(loss, ps, data, opt)
```

The objective will almost always be defined in terms of some *cost function* that measures the distance of the prediction `m(x)` from the target `y`. Flux has several of these built in, like `mse` for mean squared error or `crossentropy` for cross entropy loss, but you can calculate it however you want.
For a list of all built-in loss functions, check out the [losses reference](../models/losses.md).

At first glance it may seem strange that the model that we want to train is not part of the input arguments of `Flux.train!` too. However the target of the optimizer is not the model itself, but the objective function that represents the departure between modelled and observed data. In other words, the model is implicitly defined in the objective function, and there is no need to give it explicitly. Passing the objective function instead of the model and a cost function separately provides more flexibility, and the possibility of optimizing the calculations.

## Model parameters

The model to be trained must have a set of tracked parameters that are used to calculate the gradients of the objective function. In the [basics](../models/basics.md) section it is explained how to create models with such parameters. The second argument of the function `Flux.train!` must be an object containing those parameters, which can be obtained from a model `m` as `params(m)`.

Such an object contains a reference to the model's parameters, not a copy, such that after their training, the model behaves according to their updated values.

Handling all the parameters on a layer by layer basis is explained in the [Layer Helpers](../models/basics.md) section. Also, for freezing model parameters, see the [Advanced Usage Guide](../models/advanced.md).

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
using IterTools: ncycle
data = ncycle([(x, y)], 3)
```

It's common to load the `x`s and `y`s separately. In this case you can use `zip`:

```julia
xs = [rand(784), rand(784), rand(784)]
ys = [rand( 10), rand( 10), rand( 10)]
data = zip(xs, ys)
```

Training data can be conveniently  partitioned for mini-batch training using the [`Flux.Data.DataLoader`](@ref) type:

```julia
X = rand(28, 28, 60000)
Y = rand(0:9, 60000)
data = DataLoader(X, Y, batchsize=128) 
```

Note that, by default, `train!` only loops over the data once (a single "epoch").
A convenient way to run multiple epochs from the REPL is provided by `@epochs`.

```julia
julia> using Flux: @epochs

julia> @epochs 2 println("hello")
INFO: Epoch 1
hello
INFO: Epoch 2
hello

julia> @epochs 2 Flux.train!(...)
# Train for two epochs
```

```@docs
Flux.@epochs
```

## Callbacks

`train!` takes an additional argument, `cb`, that's used for callbacks so that you can observe the training process. For example:

```julia
train!(objective, ps, data, opt, cb = () -> println("training"))
```

Callbacks are called for every batch of training data. You can slow this down using `Flux.throttle(f, timeout)` which prevents `f` from being called more than once every `timeout` seconds.

A more typical callback might look like this:

```julia
test_x, test_y = # ... create single batch of test data ...
evalcb() = @show(loss(test_x, test_y))

Flux.train!(objective, ps, data, opt,
            cb = throttle(evalcb, 5))
```

Calling `Flux.stop()` in a callback will exit the training loop early.

```julia
cb = function ()
  accuracy() > 0.9 && Flux.stop()
end
```

## Custom Training loops

The `Flux.train!` function can be very convenient, especially for simple problems.
Its also very flexible with the use of callbacks.
But for some problems its much cleaner to write your own custom training loop.
An example follows that works similar to the default `Flux.train` but with no callbacks.
You don't need callbacks if you just code the calls to your functions directly into the loop.
E.g. in the places marked with comments.

```julia
function my_custom_train!(loss, ps, data, opt)
  # training_loss is declared local so it will be available for logging outside the gradient calculation.
  local training_loss
  ps = Params(ps)
  for d in data
    gs = gradient(ps) do
      training_loss = loss(d...)
      # Code inserted here will be differentiated, unless you need that gradient information
      # it is better to do the work outside this block.
      return training_loss
    end
    # Insert whatever code you want here that needs training_loss, e.g. logging.
    # logging_callback(training_loss)
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
    update!(opt, ps, gs)
    # Here you might like to check validation set accuracy, and break out to do early stopping.
  end
end
```

You could simplify this further, for example by hard-coding in the loss function.

Another possibility is to use [`Zygote.pullback`](https://fluxml.ai/Zygote.jl/dev/adjoints/#Pullbacks-1)
to access the training loss and the gradient simultaneously.

```julia
function my_custom_train!(loss, ps, data, opt)
  ps = Params(ps)
  for d in data
    # back is a method that computes the product of the gradient so far with its argument.
    train_loss, back = Zygote.pullback(() -> loss(d...), ps)
    # Insert whatever code you want here that needs training_loss, e.g. logging.
    # logging_callback(training_loss)
    # Apply back() to the correct type of 1.0 to get the gradient of loss.
    gs = back(one(train_loss))
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
    update!(opt, ps, gs)
    # Here you might like to check validation set accuracy, and break out to do early stopping.
  end
end
```
