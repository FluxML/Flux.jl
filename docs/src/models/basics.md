# Model Building Basics

## The Model

*... Initialising Photon Beams ...*

The core concept in Flux is the *model*. A model (or "layer") is simply a function with parameters. For example, in plain Julia code, we could define the following function to represent a logistic regression (or simple neural network):

```julia
W = randn(3,5)
b = randn(3)
affine(x) = W * x + b

x1 = rand(5) # [0.581466,0.606507,0.981732,0.488618,0.415414]
y1 = softmax(affine(x1)) # [0.32676,0.0974173,0.575823]
```

`affine` is simply a function which takes some vector `x1` and outputs a new one `y1`. For example, `x1` could be data from an image and `y1` could be predictions about the content of that image. However, `affine` isn't static. It has *parameters* `W` and `b`, and if we tweak those parameters we'll tweak the result – hopefully to make the predictions more accurate.

This is all well and good, but we usually want to have more than one affine layer in our network; writing out the above definition to create new sets of parameters every time would quickly become tedious. For that reason, we want to use a *template* which creates these functions for us:

```julia
affine1 = Affine(5, 5)
affine2 = Affine(5, 5)

softmax(affine1(x1)) # [0.167952, 0.186325, 0.176683, 0.238571, 0.23047]
softmax(affine2(x1)) # [0.125361, 0.246448, 0.21966, 0.124596, 0.283935]
```

We just created two separate `Affine` layers, and each contains its own version of `W` and `b`, leading to a different result when called with our data. It's easy to define templates like `Affine` ourselves (see [The Template](@ref)), but Flux provides `Affine` out of the box, so we'll use that for now.

## Combining Models

*... Inflating Graviton Zeppelins ...*

A more complex model usually involves many basic layers like `affine`, where we use the output of one layer as the input to the next:

```julia
mymodel1(x) = softmax(affine2(σ(affine1(x))))
mymodel1(x1) # [0.187935, 0.232237, 0.169824, 0.230589, 0.179414]
```

This syntax is again a little unwieldy for larger networks, so Flux provides another template of sorts to create the function for us:

```julia
mymodel2 = Chain(affine1, σ, affine2, softmax)
mymodel2(x2) # [0.187935, 0.232237, 0.169824, 0.230589, 0.179414]
```

`mymodel2` is exactly equivalent to `mymodel1` because it simply calls the provided functions in sequence. We don't have to predefine the affine layers and can also write this as:

```julia
mymodel3 = Chain(
  Affine(5, 5), σ,
  Affine(5, 5), softmax)
```

You now know enough to take a look at the [logistic regression](../examples/logreg.md) example, if you haven't already.

## A Function in Model's Clothing

*... Booting Dark Matter Transmogrifiers ...*

We noted above that a "model" is a function with some number of trainable parameters. This goes both ways; a normal Julia function like `exp` is effectively a model with 0 parameters. Flux doesn't care, and anywhere that you use one, you can use the other. For example, `Chain` will happily work with regular functions:

```julia
foo = Chain(exp, sum, log)
foo([1,2,3]) == 3.408 == log(sum(exp([1,2,3])))
```

## The Template

*... Calculating Tax Expenses ...*

So how does the `Affine` template work? We don't want to duplicate the code above whenever we need more than one affine layer:

```julia
W₁, b₁ = randn(...)
affine₁(x) = W₁*x + b₁
W₂, b₂ = randn(...)
affine₂(x) = W₂*x + b₂
model = Chain(affine₁, affine₂)
```

Here's one way we could solve this: just keep the parameters in a Julia type, and define how that type acts as a function:

```julia
type MyAffine
  W
  b
end

# Use the `MyAffine` layer as a model
(l::MyAffine)(x) = l.W * x + l.b

# Convenience constructor
MyAffine(in::Integer, out::Integer) =
  MyAffine(randn(out, in), randn(out))

model = Chain(MyAffine(5, 5), MyAffine(5, 5))

model(x1) # [-1.54458,0.492025,0.88687,1.93834,-4.70062]
```

This is much better: we can now make as many affine layers as we want. This is a very common pattern, so to make it more convenient we can use the `@net` macro:

```julia
@net type MyAffine
  W
  b
  x -> W * x + b
end
```

The function provided, `x -> W * x + b`, will be used when `MyAffine` is used as a model; it's just a shorter way of defining the `(::MyAffine)(x)` method above.

However, `@net` does not simply save us some keystrokes; it's the secret sauce that makes everything else in Flux go. For example, it analyses the code for the forward function so that it can differentiate it or convert it to a TensorFlow graph.

The above code is almost exactly how `Affine` is defined in Flux itself! There's no difference between "library-level" and "user-level" models, so making your code reusable doesn't involve a lot of extra complexity. Moreover, much more complex models than `Affine` are equally simple to define.

### Sub-Templates

`@net` models can contain sub-models as well as just array parameters:

```julia
@net type TLP
  first
  second
  function (x)
    l1 = σ(first(x))
    l2 = softmax(second(l1))
  end
end
```

Just as above, this is roughly equivalent to writing:

```julia
type TLP
  first
  second
end

function (self::TLP)(x)
  l1 = σ(self.first(x))
  l2 = softmax(self.second(l1))
end
```

Clearly, the `first` and `second` parameters are not arrays here, but should be models themselves, and produce a result when called with an input array `x`. The `Affine` layer fits the bill so we can instantiate `TLP` with two of them:

```julia
model = TLP(Affine(10, 20),
            Affine(20, 15))
x1 = rand(20)
model(x1) # [0.057852,0.0409741,0.0609625,0.0575354 ...
```

You may recognise this as being equivalent to

```julia
Chain(
  Affine(10, 20), σ
  Affine(20, 15), softmax)
```

given that it's just a sequence of calls. For simple networks `Chain` is completely fine, although the `@net` version is more powerful as we can (for example) reuse the output `l1` more than once.
