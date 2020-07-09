# Optimisers

Consider a [simple linear regression](../models/basics.md). We create some dummy data, calculate a loss, and backpropagate to calculate gradients for the parameters `W` and `b`.

```julia
using Flux

W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
l = loss(x, y) # ~ 3

θ = Params([W, b])
grads = gradient(() -> loss(x, y), θ)
```

We want to update each parameter, using the gradient, in order to improve (reduce) the loss. Here's one way to do that:

```julia
using Flux.Optimise: update!

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
Flux.Optimise.update!
Descent
Momentum
Nesterov
RMSProp
ADAM
RADAM
AdaMax
ADAGrad
ADADelta
AMSGrad
NADAM
ADAMW
```

## Optimiser Interface

Flux's optimisers are built around a `struct` that holds all the optimiser parameters along with a definition of how to apply the update rule associated with it. We do this via the `apply!` function which takes the optimiser as the first argument followed by the parameter and its corresponding gradient.

In this manner Flux also allows one to create custom optimisers to be used seamlessly. Let's work this with a simple example.

```julia
mutable struct Momentum
  eta
  rho
  velocity
end

Momentum(eta::Real, rho::Real) = Momentum(eta, rho, IdDict())
```

The `Momentum` type will act as our optimiser in this case. Notice that we have added all the parameters as fields, along with the velocity which we will use as our state dictionary. Each parameter in our models will get an entry in there. We can now define the rule applied when this optimiser is invoked.

```julia
function Flux.Optimise.apply!(o::Momentum, x, Δ)
  η, ρ = o.eta, o.rho
  v = get!(o.velocity, x, zero(x))::typeof(x)
  @. v = ρ * v - η * Δ
  @. Δ = -v
end
```

This is the basic definition of a Momentum update rule given by:

```math
v = ρ * v - η * Δ
w = w - v
```

The `apply!` defines the update rules for an optimiser `opt`, given the parameters and gradients. It returns the updated gradients. Here, every parameter `x` is retrieved from the running state `v` and subsequently updates the state of the optimiser.

Flux internally calls on this function via the `update!` function. It shares the API with `apply!` but ensures that multiple parameters are handled gracefully.

## Composing Optimisers

Flux defines a special kind of optimiser simply called `Optimiser` which takes in arbitrary optimisers as input. Its behaviour is similar to the usual optimisers, but differs in that it acts by calling the optimisers listed in it sequentially. Each optimiser produces a modified gradient
that will be fed into the next, and the resultant update will be applied to the parameter as usual. A classic use case is where adding decays is desirable. Flux defines some basic decays including `ExpDecay`, `InvDecay` etc.

```julia
opt = Optimiser(ExpDecay(0.001, 0.1, 1000, 1e-4), Descent())
```

Here we apply exponential decay to the `Descent` optimiser. The defaults of `ExpDecay` say that its learning rate will be decayed every 1000 steps.
It is then applied like any optimiser.

```julia
w = randn(10, 10)
w1 = randn(10,10)
ps = Params([w, w1])

loss(x) = Flux.Losses.mse(w * x, w1 * x)

loss(rand(10)) # around 9

for t = 1:10^5
  θ = Params([w, w1])
  θ̄ = gradient(() -> loss(rand(10)), θ)
  Flux.Optimise.update!(opt, θ, θ̄)
end

loss(rand(10)) # around 0.9
```

In this manner it is possible to compose optimisers for some added flexibility.

## Decays

Similar to optimisers, Flux also defines some simple decays that can be used in conjunction with other optimisers, or standalone.

```@docs
ExpDecay
InvDecay
WeightDecay
```

## Gradient Clipping

Gradient clipping is useful for training recurrent neural networks, which have a tendency to suffer from the exploding gradient problem. An example usage is

```julia
opt = Optimiser(ClipValue(1e-3), ADAM(1e-3))
```

```@docs
ClipValue
ClipNorm
```