# [Training](@id man-training)

Training refers to the process of slowly adjusting the parameters of a model to make it work better.
Besides the model itself, we will need three things:

* An *objective function* that evaluates how well a model is doing on some input.
* An *optimisation rule* which describes how the model's parameters should be adjusted.
* Some *training data* to use as the input during this process.

Usually the training data is some collection of examples (or batches of examples) which
are handled one-by-one. One *epoch* of training means that each example is used once,
something like this:

```julia
for data in train_set
  # Unpack this datapoint into the input and the
  # desired result (for "supervised" training):
  input, label = data

  # Calculate the gradient of the objective
  # with respect to the parameters within the model:
  grads = Flux.gradient(model) do m
      result = m(input)
      loss(result, label)
  end

  # Update the parameters so as to reduce the objective,
  # according to a particular optimiser:
  Flux.update!(opt, model, grads[1])
end
```

This isn't pseudo-code, but is precisely how traning is done.
This loop can also be written using the function [`train!`](@ref Train.train!),
but it's helpful to undersand the pieces first:

```julia
train!(model, train_set, opt) do m, x, y
  loss(m(x), y)
end
```

## Model Gradients

Fist recall from the section on [taking gradients](@ref man-taking-gradients) that 
`Flux.gradient(f, a, b)` always calls `f(a, b)`, and returns a tuple `(∂f_∂a, ∂f_∂b)`.
In the code above, the function `f` is an anonymous function with one argument,
created by the `do` block, hence  `grads` is a tuple with one element.
Instead of a `do` block, we could have written:

```julia
grads = Flux.gradient(m -> loss(m(input), label), model)
```

Since the model is some nested set of layers, `grads[1]` is a similarly nested set of
`NamedTuple`s, ultimately containing gradient components. These matching tree-like
structures are what Zygote calls "explicit" gradients.

It is important that the execution of the model takes place inside the call to `gradient`,
in order for the influence of the model's parameters to be observed by Zygote.

!!! note "Explicit vs implicit gradients"
    Flux used to use Zygote's "implicit" mode, which looks like this:
    ```
    pars = Flux.params(model)
    grad = Flux.gradient(() -> loss(model(input), label), pars)
    ```
    Here `pars::Params` and `grad::Grads` are two dictionary-like structures.


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
are available from the [`Flux.Losses`](../models/losses.md) module.

!!! note "Implicit-style loss functions"
    Flux used to need a loss function which closed over a reference to the model,
    instead of being a pure function. Thus in old code you may see something like
    ```
    loss(x, y) = sum((model(x) .- y).^2)
    ```
    which defines a function making reference to a particular global variable `model`.
    This is no longer the preferred style.

## Optimisation Rules

The simplest kind of optimisation using the gradient is termed *gradient descent*
(or sometimes *stochastic gradient descent* when it is applied to individual examples
in a loop, not to the entire dataset at once).

This needs a *learning rate* which is a small number describing how fast to walk downhill,
usually written as the Greek letter "eta", `η`. 

```julia
η = 0.01   # learning rate

# For each parameter array, update
# according to the corresponding gradient:
fmap(model, grads[1]) do p, g
  p .= p .- η .* g
end
```

This is wrapped up as a function [`update!`](@ref Flux.Optimise.update!), which can be used as follows:

```julia
Flux.update!(Descent(0.01), model, grads[1])
```

There are many other optimisation rules, which adjust the step size and direction.
Most require some memory of the gradients from earlier steps. The function [`setup`](@ref Flux.Train.setup)
creates the necessary storage for this, for a particular model. This should be done
once, before training, and looks like this: 

```julia
# Initialise momentum 
opt = Flux.setup(Adam(0.001), model)

for data in train_set
  ...  

  # 
  Flux.update!(opt, model, grads[1])
end
```

Many commonly used optimisation rules, such as [`Adam`](@ref Flux.Optimise.Adam), are built-in.
These are listed on the [optimisers](@ref man-optimisers) page.


!!! note "Implicit-style optimiser state"
    This `setep` makes another tree-like structure. Old versions of Flux did not do this,
    and instead stored a dictionary-like structure within the optimiser `Adam(0.001)`.
    This was initialised on first use of the version of `update!` for "implicit" parameters.


## Datasets

The loop above iterates through `train_set`, expecting at each step a tuple `(input, label)`.
The very simplest such object is a vector of tuples, such as this:

```julia
x = randn(28, 28)
y = rand(10)
data = [(x, y)]
```

or `data = [(x, y), (x, y), (x, y)]` for the same values three times.

To get data into this format, you might want `zip` to combine a list of different `x`s
with a list of different `y`s:

```julia
xs = [rand(28, 28), rand(28, 28), rand(28, 28)]
ys = [rand(10), rand(10), rand(10)]
data = zip(xs, ys)

first(data) isa Tuple{Matrix, Vector}  # true
```

Very often, the initial data is large arrays which you need to slice into examples:

```julia
X = rand(28, 28, 60_000)
Y = rand(10, 60_000)
data = zip(eachslice(X; dims=3), eachcol(Y))

first(data) isa Tuple{Matrix, Vector}  # true
```

Here each iteration will use one matrix `x` (an image, perhaps) and one vector `y`.
It is very common to instead train on *batches* of such inputs (or *mini-batches*,
the two words mean the same thing) both for efficiency and for better results.
This can be easily done using the [`DataLoader`](@ref Flux.Data.DataLoader):

```julia
data = Flux.DataLoader((X, Y), batchsize=32)

x1, y1 = first(data)
size(x1) == (28, 28, 32)
length(data) == 1875 === 60_000 ÷ 32
```

Flux's layers are set up to accept such a batch of input data,
and the convolutional layers such as [Conv](@ref Flux.Conv) require it.


## Training Loops

Very simple training loops like the one above can be written compactly using
the [`train!`](@ref) function. Including `setup`, this reads:

```julia
opt = Flux.setup(Adam(), model)

train!(model, train_set, opt) do m, x, y
  loss(m(x), y)
end
```

!!! note "Implicit-style `train!`"
    This is the "explicit" method of `train!`, which takes the result of `setup` as its 4th argument.
    The 1st argument (from the `do` block) is a function which accepts the model itself.
    Old Flux versions provided a method of `train!` for "implicit" parameters,
    which works like this:
    ```
    train!((x,y) -> loss(model(x), y), Flux.params(model), train_set, Adam())
    ```

Real training loops often need more flexibility, and the best way to do this is just
to write the loop. This is ordinary Julia code, without any need to work through some
callback API. Here is an example, in which it may be helpful to note:

* The function [`withgradient`](@ref Zygote.withgradient) is like `gradient` but also
  returns the value of the function, for logging or diagnostic use.
* Logging or printing is best done outside of the `gradient` call,
  as there is no need to differentiate these commands.
* Julia's `break` and `continue` keywords let you exit from parts of the loop.

```julia
opt = Flux.setup(Adam(), model)

log = []
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

    Flux.update!(opt, model, grads[1])
  end

  # Compute some accuracy, and save details to log
  acc = my_accuracy(model, train_set)
  push!(log, (; acc, losses))

  # Stop training when some criterion is reached
  if  acc > 0.95
    println("stopping after $epoch epochs")
    break
  end
end
```


## Implicit vs Explicit

Flux used to handle gradients, training, and optimisation rules quite differently.
The new style described above is called "explicit" by Zygote, and the old style "implicit".
Flux 0.13 is the transitional version which supports both.

For full details on the implicit style, see [Flux 0.13.6 documentation](https://fluxml.ai/Flux.jl/v0.13.6/training/training/).

