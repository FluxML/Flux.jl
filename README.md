# Флукс

## What?

Flux is a high-level API for machine learning, implemented in Julia. It could be seen has Julia's answer to Keras, though it has its own ideas, quirks and possibilities.

The current focus is on ANNs with TensorFlow as a backend. While it's in a very early working-prototype stage, you can see what works so far in the [examples folder](/examples).

Flux's approach is heavily declarative (as opposed to imperative graph building, which has all the drawbacks of writing one huge macro). Models can be described in high-level terms close to the mathematical notation. Model descriptions are be separated from their implementations, and changes to the way a model is used should never require changes to the model itself.

A core idea is that the same problems that come up when building ML models (composition and reuse, variable scoping, applying optimisations etc.) have already been solved by programming languages. If we think of building models as programming, we can reuse those solutions, greatly reducing the barriers to learning and using ML systems.

There are also some unusual possibilities enabled by Julia itself. Julia's speed makes it trivial to prototype fully custom layers with reasonable performance. In future, GPU codegen may enable us to hook custom layers into TensorFlow and other backends.

## How?

We can describe simple models through a convenient interface:

```julia
m = Chain(
  Input(784),
  Dense(128), relu,
  Dense( 64), relu,
  Dense( 10), softmax)
```

Models are simple functions with state, so we can immediately see what the network does:

```julia
m(randn(784)) #> [0.101, 0.101, 0.099, 0.100, ...]
```

What if we need a custom layer? Here's one equivalent to `Dense` above:

```julia
# Simple Julia type with two fields – @net defines some extra methods like the
# backward pass
@net type FullyConnected
  W; b
  x -> x*W + b
end

# Convenience constructor, initialise the parameters with random weights
FullyConnected(in::Integer, out::Integer) = FullyConnected(randn(out, in), randn(out))

foo = FullyConnected(10, 5)
foo(randn(10)) #> [0.00981148,0.0456825,...]
```

We can already insert this model into combining models like `Chain`. If you want to use the layer more than once or make a copy, just do so; no variable scoping here. We can also define models that contain other models:

```julia
@net type Perceptron
  layer
  x -> σ(layer(x))
end

Perceptron(in, out) = Perceptron(Dense(in, out))
```

This defines a simple perceptron layer which we can use in the same way as `Dense` above. We can draw arbitrary graphs, including those with splits, combines or recurrences, in a fully declarative way *[this API is a WIP]*:

```julia
@net type SimpleRecurrent
  Wx; Wh; b
  hidden
  function (x)
    hidden = σ(Wx * x + Wh * hidden + b)
  end
end
```

`hidden`'s dependence on itself creates a cycle, and wherever `hidden` is used, that value will come from the last run of the network. We can also define this same layer as a composition of others:

```julia
@net type SimpleRecurrent2
  layer
  hidden
  function (x)
    hidden = σ(layer(vcat(x, hidden)))
  end
end
```

Though further from the equations, this has the advantage of further reuse and customizability. For example, `layer` could be a simple `Dense(x, y)` as before or it could be a `Dropout(Dense(x, y))` in order to add dropout to the recurrent layer.

When it comes time to train the model, we have a number of options for tweaking its implementation, like the backend used or unrolling settings. In Flux this is as simple as calling some functions on the original model:

```julia
model = unroll(model, 10) # Statically unroll the model
model = tf(model) # Convert the model to TensorFlow

Flux.train!(model, ...)
```
