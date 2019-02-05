# Optimisers

Consider a [simple linear regression](../models/basics.md). We create some dummy data, calculate a loss, and backpropagate to calculate gradients for the parameters `W` and `b`.

```julia
using Flux, Flux.Tracker

W = param(rand(2, 5))
b = param(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
l = loss(x, y) # ~ 3

θ = Params([W, b])
grads = Tracker.gradient(() -> loss(x, y), θ)
```

We want to update each parameter, using the gradient, in order to improve (reduce) the loss. Here's one way to do that:

```julia
using Flux.Tracker: grad, update!

η = 0.1 # Learning Rate
for p in (W, b)
  update!(p, -η * grads[p])
end
```

Running this will alter the parameters `W` and `b` and our loss should go down. Flux provides a more general way to do optimiser updates like this.

```julia
opt = Descent(0.1) # Gradient descent with learning rate 0.1

for p in (W, b)
  update!(opt, p, grads[p])
end
```

An optimiser `update!` accepts a parameter and a gradient, and updates the parameter according to the chosen rule. We can also pass `opt` to our [training loop](training.md), which will update all parameters of the model in a loop. However, we can now easily replace `Descent` with a more advanced optimiser such as `ADAM`.

## Optimiser Reference

All optimisers return an object that, when passed to `train!`, will update the parameters passed to it.

```@docs
Descent
Momentum
Nesterov
ADAM
```
