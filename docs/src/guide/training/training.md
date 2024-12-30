# [Training a Flux Model](@id man-training)

Training refers to the process of slowly adjusting the parameters of a model to make it work better.
Besides the model itself, we will need three things:

* An *objective function* that evaluates how well a model is doing on some input.
* An *optimisation rule* which describes how the model's parameters should be adjusted.
* Some *training data* to use as the input during this process.

Usually the training data is some collection of examples (or batches of examples) which
are handled one-by-one. One *epoch* of training means that each example is used once,
something like this:

```julia
# Initialise the optimiser for this model:
opt_state = Flux.setup(rule, model)

for data in train_set
  # Unpack this element (for supervised training):
  input, label = data

  # Calculate the gradient of the objective
  # with respect to the parameters within the model:
  grads = Flux.gradient(model) do m
      result = m(input)
      loss(result, label)
  end

  # Update the parameters so as to reduce the objective,
  # according the chosen optimisation rule:
  Flux.update!(opt_state, model, grads[1])
end
```

This loop can also be written using the function [`train!`](@ref Flux.Train.train!),
but it's helpful to understand the pieces first:

```julia
train!(model, train_set, opt_state) do m, x, y
  loss(m(x), y)
end
```

## Model Gradients

Fist recall from the section on [taking gradients](@ref man-taking-gradients) that
`Flux.gradient(f, a, b)` always calls `f(a, b)`, and returns a tuple `(∂f_∂a, ∂f_∂b)`.
In the code above, the function `f` passed to `gradient` is an anonymous function with
one argument, created by the `do` block, hence  `grads` is a tuple with one element.
Instead of a `do` block, we could have written:

```julia
grads = Flux.gradient(m -> loss(m(input), label), model)
```

Since the model is some nested set of layers, `grads[1]` is a similarly nested set of
`NamedTuple`s, ultimately containing gradient components. If (for example) 
`θ = model.layers[1].weight[2,3]` is one scalar parameter, an entry in a matrix of weights,
then the derivative of the loss with respect to it is `∂f_∂θ = grads[1].layers[1].weight[2,3]`.

It is important that the execution of the model takes place inside the call to `gradient`,
in order for the influence of the model's parameters to be observed by Zygote.

It is also important that every `update!` step receives a newly computed gradient,
as it will change whenever the model's parameters are changed, and for each new data point.


## Loss Functions

The objective function must return a number representing how far the model is from
the desired result. This is termed the *loss* of the model.

This number can be produced by any ordinary Julia code, but this must be executed
within the call to `gradient`. For instance, we could define a function
```julia
loss(y_hat, y) = sum((y_hat .- y).^2)
```
or write this directly inside the `do` block above. Many commonly used functions,
like [`mse`](@ref Flux.Losses.mse) for mean-squared error or [`crossentropy`](@ref Flux.Losses.crossentropy) for cross-entropy loss,
are available from the [`Flux.Losses`](../../reference/models/losses.md) module.


## Optimisation Rules

The simplest kind of optimisation using the gradient is termed *gradient descent*
(or sometimes *stochastic gradient descent* when, as here, it is not applied to the entire dataset at once).

Gradient descent needs a *learning rate* which is a small number describing how fast to walk downhill,
usually written as the Greek letter "eta", `η`. This is often described as a *hyperparameter*,
to distinguish it from the parameters which are being updated `θ = θ - η * ∂loss_∂θ`.
We want to update all the parameters in the model, like this:

```julia
η = 0.01   # learning rate

# For each parameter array, update
# according to the corresponding gradient:
fmap(model, grads[1]) do p, g
  p .= p .- η .* g
end
```

A slightly more refined version of this loop to update all the parameters is wrapped up as a function [`update!`](@ref Optimisers.update!)`(opt_state, model, grads[1])`.
And the learning rate is the only thing stored in the [`Descent`](@ref Optimisers.Descent) struct.

However, there are many other optimisation rules, which adjust the step size and
direction in various clever ways.
Most require some memory of the gradients from earlier steps, rather than always
walking straight downhill -- [`Momentum`](@ref Optimisers.Momentum) is the simplest.
The function [`setup`](@ref Flux.Train.setup) creates the necessary storage for this, for a particular model.
It should be called once, before training, and returns a tree-like object which is the
first argument of `update!`. Like this:

```julia
# Initialise momentum 
opt_state = Flux.setup(Momentum(0.01, 0.9), model)

for data in train_set
  grads = [...]

  # Update both model parameters and optimiser state:
  Flux.update!(opt_state, model, grads[1])
end
```

Many commonly-used optimisation rules, such as [`Adam`](@ref Optimisers.Adam), are built-in.
These are listed on the [optimisers](@ref man-optimisers) page.

!!! compat "Implicit-style optimiser state"
    This `setup` makes another tree-like structure. Old versions of Flux did not do this,
    and instead stored a dictionary-like structure within the optimiser `Adam(0.001)`.
    This was initialised on first use of the version of `update!` for "implicit" parameters.


## Datasets & Batches

The loop above iterates through `train_set`, expecting at each step a tuple `(input, label)`.
The very simplest such object is a vector of tuples, such as this:

```julia
x = randn(28, 28)
y = rand(10)
data = [(x, y)]
```

or `data = [(x, y), (x, y), (x, y)]` for the same values three times.

Very often, the initial data is large arrays which you need to slice into examples.
To produce one iterator of pairs `(x, y)`, you might want `zip`:

```julia
X = rand(28, 28, 60_000);  # many images, each 28 × 28
Y = rand(10, 60_000)
data = zip(eachslice(X; dims=3), eachcol(Y))

first(data) isa Tuple{AbstractMatrix, AbstractVector}  # true
```

Here each iteration will use one matrix `x` (an image, perhaps) and one vector `y`.
It is very common to instead train on *batches* of such inputs (or *mini-batches*,
the two words mean the same thing) both for efficiency and for better results.
This can be easily done using the [`DataLoader`](@ref Flux.DataLoader):

```julia
data = Flux.DataLoader((X, Y), batchsize=32)

x1, y1 = first(data)
size(x1) == (28, 28, 32)
length(data) == 1875 === 60_000 ÷ 32
```

Flux's layers are set up to accept such a batch of input data,
and the convolutional layers such as [`Conv`](@ref Flux.Conv) require it.
The batch index is always the last dimension.

## Training Loops

Simple training loops like the one above can be written compactly using
the [`train!`](@ref Flux.Train.train!) function. Including `setup`, this reads:

```julia
opt_state = Flux.setup(Adam(), model)

for epoch in 1:100
  Flux.train!(model, train_set, opt_state) do m, x, y
    loss(m(x), y)
  end
end
```

Or explicitly writing the anonymous function which this `do` block creates,
`train!((m,x,y) -> loss(m(x),y), model, train_set, opt_state)` is exactly equivalent.

Real training loops often need more flexibility, and the best way to do this is just
to write the loop. This is ordinary Julia code, without any need to work through some
callback API. Here is an example, in which it may be helpful to note:

* The function [`withgradient`](@ref Zygote.withgradient) is like `gradient` but also
  returns the value of the function, for logging or diagnostic use.
* Logging or printing is best done outside of the `gradient` call,
  as there is no need to differentiate these commands.
* To use `result` for logging purposes, you could change the `do` block to end with 
  `return my_loss(result, label), result`, i.e. make the function passed to `withgradient`
  return a tuple. The first element is always the loss.
* Julia's `break` and `continue` keywords let you exit from parts of the loop.

```julia
opt_state = Flux.setup(Adam(), model)

my_log = []
for epoch in 1:100
  losses = Float32[]
  for (i, data) in enumerate(train_set)
    input, label = data

    val, grads = Flux.withgradient(model) do m
      # Any code inside here is differentiated.
      # Evaluation of the model and loss must be inside!
      result = m(input)
      my_loss(result, label)
    end

    # Save the loss from the forward pass. (Done outside of gradient.)
    push!(losses, val)

    # Detect loss of Inf or NaN. Print a warning, and then skip update!
    if !isfinite(val)
      @warn "loss is $val on item $i" epoch
      continue
    end

    Flux.update!(opt_state, model, grads[1])
  end

  # Compute some accuracy, and save details as a NamedTuple
  acc = my_accuracy(model, train_set)
  push!(my_log, (; acc, losses))

  # Stop training when some criterion is reached
  if  acc > 0.95
    println("stopping after $epoch epochs")
    break
  end
end
```

## Regularisation

The term *regularisation* covers a wide variety of techniques aiming to improve the
result of training. This is often done to avoid overfitting.

Some of these can be implemented by simply modifying the loss function.
*L₂ regularisation* (sometimes called ridge regression) adds to the loss a penalty
proportional to `θ^2` for every scalar parameter.
A very simple model could be implemented as follows:

```julia
grads = Flux.gradient(densemodel) do m
  result = m(input)
  penalty = sum(abs2, m.weight)/2 + sum(abs2, m.bias)/2
  my_loss(result, label) + 0.42f0 * penalty
end
```

Accessing each individual parameter array by hand won't work well for large models.
Instead, we can use [`Flux.trainables`](@ref Optimisers.trainables) to collect all of them,
and then apply a function to each one, and sum the result:

```julia
pen_l2(x::AbstractArray) = sum(abs2, x)/2

grads = Flux.gradient(model) do m
  result = m(input)
  penalty = sum(pen_l2, Flux.trainables(m))
  my_loss(result, label) + 0.42f0 * penalty
end
```

However, the gradient of this penalty term is very simple: It is proportional to the original weights.
So there is a simpler way to implement exactly the same thing, by modifying the optimiser
instead of the loss function. This is done by replacing this:

```julia
opt_state = Flux.setup(Adam(0.1), model)
```

with this:

```julia
decay_opt_state = Flux.setup(OptimiserChain(WeightDecay(0.42), Adam(0.1)), model)
```

Flux's optimisers are really modifications applied to the gradient before using it to update
the parameters, and [`OptimiserChain`](@ref Optimisers.OptimiserChain) applies two such modifications.
The first, [`WeightDecay`](@ref Optimisers.WeightDecay) adds `0.42` times the original parameter to the gradient,
matching the gradient of the penalty above (with the same, unrealistically large, constant).
After that, in either case, [`Adam`](@ref Optimisers.Adam) computes the final update.

The same trick works for *L₁ regularisation* (also called Lasso), where the penalty is 
`pen_l1(x::AbstractArray) = sum(abs, x)` instead. This is implemented by `SignDecay(0.42)`.

The same `OptimiserChain` mechanism can be used for other purposes, such as gradient clipping with [`ClipGrad`](@ref Optimisers.ClipGrad) or [`ClipNorm`](@ref Optimisers.ClipNorm).

Besides L1 / L2 / weight decay, another common and quite different kind of regularisation is
provided by the [`Dropout`](@ref Flux.Dropout) layer. This turns off some outputs of the
previous layer during training.
It should switch automatically, but see [`trainmode!`](@ref Flux.trainmode!) / [`testmode!`](@ref Flux.testmode!) to manually enable or disable this layer.

## Learning Rate Schedules

Finer control of training, you may wish to alter the learning rate mid-way through training.
This can be done with [`adjust!`](@ref Flux.adjust!), like this:

```julia
opt_state = Flux.setup(Adam(0.1), model)  # initialise once

for epoch in 1:1000
  train!([...], state)  # Train with η = 0.1 for first 100,
  if epoch == 100       # then change to use η = 0.01 for the rest.
    Flux.adjust!(opt_state, 0.01)
  end
end
```

Other hyper-parameters can also be adjusted, such as `Flux.adjust!(opt_state, beta = (0.8, 0.99))`.
And such modifications can be applied to just one part of the model.
For instance, this sets a different learning rate for the encoder and the decoder:

```julia
# Consider some model with two parts:
bimodel = Chain(enc = [...], dec = [...])

# This returns a tree whose structure matches the model:
opt_state = Flux.setup(Adam(0.02), bimodel)

# Adjust the learning rate to be used for bimodel.layers.enc
Flux.adjust!(opt_state.layers.enc, 0.03)
```


## Scheduling Optimisers

In practice, it is fairly common to schedule the learning rate of an optimiser to obtain faster convergence. There are a variety of popular scheduling policies, and you can find implementations of them in [ParameterSchedulers.jl](http://fluxml.ai/ParameterSchedulers.jl/stable). The documentation for ParameterSchedulers.jl provides a more detailed overview of the different scheduling policies, and how to use them with Flux optimisers. Below, we provide a brief snippet illustrating a [cosine annealing](https://arxiv.org/pdf/1608.03983.pdf) schedule with a momentum optimiser.

First, we import ParameterSchedulers.jl and initialize a cosine annealing schedule to vary the learning rate between `1e-4` and `1e-2` every 10 epochs. We also create a new [`Momentum`](@ref Optimisers.Momentum) optimiser.
```julia
using ParameterSchedulers

opt_state = Flux.setup(Momentum(), model)
schedule = Cos(λ0 = 1e-4, λ1 = 1e-2, period = 10)
for (eta, epoch) in zip(schedule, 1:100)
  Flux.adjust!(opt_state, eta)
  # your training code here
end
```
`schedule` can also be indexed (e.g. `schedule(100)`) or iterated like any iterator in Julia.

ParameterSchedulers.jl schedules are stateless (they don't store their iteration state). If you want a _stateful_ schedule, you can use `ParameterSchedulers.Stateful`:
```julia
using ParameterSchedulers: Stateful, next!

schedule = Stateful(Cos(λ0 = 1e-4, λ1 = 1e-2, period = 10))
for epoch in 1:100
  Flux.adjust!(opt_state, next!(schedule))
  # your training code here
end
```

Finally, a scheduling function can be incorporated into the optimser's state, advanced at each gradient update step, and possibly passed to the `train!` function. See [this section](https://fluxml.ai/ParameterSchedulers.jl/stable/tutorials/optimizers/#Working-with-Flux-optimizers) of ParameterSchedulers.jl documentation for more details.

ParameterSchedulers.jl allows for many more scheduling policies including arbitrary functions, looping any function with a given period, or sequences of many schedules. See the [ParameterSchedulers.jl documentation](https://fluxml.ai/ParameterSchedulers.jl/stable) for more info.

## Freezing layer parameters

To completely disable training of some part of the model, use [`freeze!`](@ref Flux.freeze!).
This is a temporary modification, reversed by `thaw!`:

```julia
Flux.freeze!(opt_state.layers.enc)

# Now training won't update parameters in bimodel.layers.enc
train!(loss, bimodel, data, opt_state)

# Un-freeze the entire model:
Flux.thaw!(opt_state)
```

While `adjust!` and `freeze!`/`thaw!` make temporary modifications to the optimiser state,
permanently removing some fields of a new layer type from training is usually done
when defining the layer, by calling for example [`@layer`](@ref Flux.@layer)` NewLayer trainable=(weight,)`.

