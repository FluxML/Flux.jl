# Model-Building Basics

## Taking Gradients

Consider a simple linear regression, which tries to predict an output array `y` from an input `x`. (It's a good idea to follow this example in the Julia repl.)

```julia
W = rand(2, 5)
b = rand(2)

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
loss(x, y) # ~ 3
```

To improve the prediction we can take the gradients of `W` and `b` with respect to the loss function and perform gradient descent. We could calculate gradients by hand, but Flux will do it for us if we tell it that `W` and `b` are trainable *parameters*.

```julia
using Flux.Tracker

W = param(W)
b = param(b)

l = loss(x, y)

back!(l)
```

`loss(x, y)` returns the same number, but it's now a *tracked* value that records gradients as it goes along. Calling `back!` then accumulates the gradient of `W` and `b`. We can see what this gradient is, and modify `W` to train the model.

```julia
using Flux.Tracker: grad, update!

Δ = grad(W)

# Update the parameter and reset the gradient
update!(W, -0.1Δ)

loss(x, y) # ~ 2.5
```

The loss has decreased a little, meaning that our prediction `x` is closer to the target `y`. If we have some data we can already try [training the model](../training/training.md).

All deep learning in Flux, however complex, is a simple generalisation of this example. Of course, models can *look* very different – they might have millions of parameters or complex control flow, and there are ways to manage this complexity. Let's see what that looks like.

## Building Layers

It's common to create more complex models than the linear regression above. For example, we might want to have two linear layers with a nonlinearity like [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) (`σ`) in between them. In the above style we could write this as:

```julia
W1 = param(rand(3, 5))
b1 = param(rand(3))
layer1(x) = W1 * x .+ b1

W2 = param(rand(2, 3))
b2 = param(rand(2))
layer2(x) = W2 * x .+ b2

model(x) = layer2(σ.(layer1(x)))

model(rand(5)) # => 2-element vector
```

This works but is fairly unwieldy, with a lot of repetition – especially as we add more layers. One way to factor this out is to create a function that returns linear layers.

```julia
function linear(in, out)
  W = param(randn(out, in))
  b = param(randn(out))
  x -> W * x .+ b
end

linear1 = linear(5, 3) # we can access linear1.W etc
linear2 = linear(3, 2)

model(x) = linear2(σ.(linear1(x)))

model(x) # => 2-element vector
```

Another (equivalent) way is to create a struct that explicitly represents the affine layer.

```julia
struct Affine
  W
  b
end

Affine(in::Integer, out::Integer) =
  Affine(param(randn(out, in)), param(randn(out)))

# Overload call, so the object can be used as a function
(m::Affine)(x) = m.W * x .+ m.b

a = Affine(10, 5)

a(rand(10)) # => 5-element vector
```

Congratulations! You just built the `Dense` layer that comes with Flux. Flux has many interesting layers available, but they're all things you could have built yourself very easily.

(There is one small difference with `Dense` – for convenience it also takes an activation function, like `Dense(10, 5, σ)`.)

## Stacking It Up

It's pretty common to write models that look something like:

```julia
layer1 = Dense(10, 5, σ)
# ...
model(x) = layer3(layer2(layer1(x)))
```

For long chains, it might be a bit more intuitive to have a list of layers, like this:

```julia
using Flux

layers = [Dense(10, 5, σ), Dense(5, 2), softmax]

model(x) = foldl((x, m) -> m(x), x, layers)

model(rand(10)) # => 2-element vector
```

Handily, this is also provided for in Flux:

```julia
model2 = Chain(
  Dense(10, 5, σ),
  Dense(5, 2),
  softmax)

model2(rand(10)) # => 2-element vector
```

This quickly starts to look like a high-level deep learning library; yet you can see how it falls out of simple abstractions, and we lose none of the power of Julia code.

A nice property of this approach is that because "models" are just functions (possibly with trainable parameters), you can also see this as simple function composition.

```julia
m = Dense(5, 2) ∘ Dense(10, 5, σ)

m(rand(10))
```

Likewise, `Chain` will happily work with any Julia function.

```julia
m = Chain(x -> x^2, x -> x+1)

m(5) # => 26
```

## Layer helpers

Flux provides a set of helpers for custom layers, which you can enable by calling

```julia
Flux.treelike(Affine)
```

This enables a useful extra set of functionality for our `Affine` layer, such as [collecting its parameters](../training/optimisers.md) or [moving it to the GPU](../gpu.md).
