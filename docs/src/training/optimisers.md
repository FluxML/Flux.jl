# Optimisers

Consider a [simple linear regression](../models/basics.md). We create some dummy data, calculate a loss, and backpropagate to calculate gradients for the parameters `W` and `b`.

```julia
using Flux.Tracker

W = param(rand(2, 5))
b = param(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
l = loss(x, y) # ~ 3

params = Params([W, b])
grads = Tracker.gradient(() -> loss(x, y), params)
```

We want to update each parameter, using the gradient, in order to improve (reduce) the loss. Here's one way to do that:

```julia
using Flux.Tracker: grad, update!

struct sgd()
  η = 0.1 # Learning Rate
  for p in (W, b)
    update!(p, -η * grads[p])
  end
end
```

If we call `sgd`, the parameters `W` and `b` will change and our loss should go down.

There are two pieces here: one is that we need a list of trainable parameters for the model (`[W, b]` in this case), and the other is the update step. In this case the update is simply gradient descent (`x .-= η .* Δ`), but we might choose to do something more advanced, like adding momentum.

In this case, getting the variables is trivial, but you can imagine it'd be more of a pain with some complex stack of layers.

```julia
m = Chain(
  Dense(10, 5, σ),
  Dense(5, 2), softmax)
```

Instead of having to write `[m[1].W, m[1].b, ...]`, Flux provides a params function `params(m)` that returns a list of all parameters in the model for you.

For the update step, there's nothing whatsoever wrong with writing the loop above – it'll work just fine – but Flux provides various *optimisers* that make it more convenient.

```julia
opt = Descent(0.1) # Gradient descent with learning rate 0.1

update!(opt, params(m)) # Carry out the update, modifying `W` and `b`.
```

An optimiser takes a parameter list and returns a function that does the same thing as `update` above. We can pass either `opt` or `update` to our [training loop](training.md), which will then run the optimiser after every mini-batch of data.

## Optimiser Reference

All optimisers return a `struct` that, when called with their `update!`, will update the parameters passed to it.

* [Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
* [Momentum](https://arxiv.org/abs/1712.09677)
* [Nesterov](https://arxiv.org/abs/1607.01981)
* [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
* [ADAM](https://arxiv.org/abs/1412.6980v8)
* [AdaMax](https://arxiv.org/abs/1412.6980v9)
* [ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
* [ADADelta](http://arxiv.org/abs/1212.5701)
* [AMSGrad](https://openreview.net/forum?id=ryQu7f-RZ)
* [NADAM](http://cs229.stanford.edu/proj2015/054_report.pdf)
* [ADAMW](https://arxiv.org/abs/1711.05101)
* InvDecay
* ExpDecay
* WeightDecay

## Optimiser API

All optimsers now exist as their own `structs` which house all the different parameters required to satisfy their respective update rules.
This is done by overloading the `Flux.Optimise.update!` method which takes the optimiser, the data and the gradients of the parameters to return the change (or the step) from the update. This follows the following design:

```julia
mutable struct Descent
  eta::Float64
end

function update!(o::Descent, x, Δ)
  Δ .*= o.eta
end
```

After this, it is sufficient to either call `Flux.train!` as usual or `Optimise.update!(opt, params(model))` in a training loop. This also comes with the change in the API of the training loop to take in the model parameters as necessary.

The `struct`s allow for decoupling the optimiser structure from its update rule allowing us to treat them as independent entities. It means we can do things like changing the optimiser parameters at will, and hooking together custom optimizers, with or without the predefined ones.

```julia
opt = Descent(0.5)
update!(opt, params(model))
opt.eta = 0.2 # valid statment, useful for annealing/ scaling
```

The `ExpDecay` function defined within Flux, takes advantage of this flexibility. It can be used as a way of scheduling the learning rate. It makes it easy to scale the learning rate, every `n` epochs. Additionaly, it is easy to specify a `clip` or a bound to the learning rate, beyond which it will be maintained throughout the remainder of the training.

```julia
ExpDecay(opt = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4)
```

The above would take the initial learning rate `0.001`, and decay it by `0.1` every `1000` steps until it reaches a minimum of `1e-4`. It can be used such that it can be applied on to any optimiser like so:

```julia
Optimiser(ExpDecay(...), Descent(...))
```

## Optimiser

An equally easy to use interface is that of `Optimiser` which is designed for creating compound optimisers or in general let us take an action against the training loop as defined on the parameters. The `update!` API remains unified.

```julia
opt1 = Descent()
opt2 = Optimiser(InvDecay(), RMSProp())
opt = Opitmiser(opt1, opt2)

update!(opt, params(model))
```

`opt = Optimiser(ExpDecay(), ADAM())` generates an optimiser that applies the previously discussed `ExpDecay` on the `ADAM` optimiser, during the training. It can also be extended as `Optimiser(..., Optimiser(...))` to create sophisticated and general optimisers that can be customised extensively. It follows many of julia's semantics, so it is possible to `push!` to them, index on them, slice them etc.