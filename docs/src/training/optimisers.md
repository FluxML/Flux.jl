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

function sgd()
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
opt = SGD([W, b], 0.1) # Gradient descent with learning rate 0.1

opt() # Carry out the update, modifying `W` and `b`.
```

An optimiser takes a parameter list and returns a function that does the same thing as `update` above. We can pass either `opt` or `update` to our [training loop](training.md), which will then run the optimiser after every mini-batch of data.

## Optimiser Reference

All optimisers return a function that, when called, will update the parameters passed to it.

```@docs
SGD
Momentum
Nesterov
ADAM
```
