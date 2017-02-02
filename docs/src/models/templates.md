# Model Templates

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

## Models in templates

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

## Constructors

`Affine` has two array parameters, `W` and `b`. Just like any other Julia type, it's easy to instantiate an `Affine` layer with parameters of our choosing:

```julia
a = Affine(rand(10, 20), rand(20))
```

However, for convenience and to avoid errors, we'd probably rather specify the input and output dimension instead:

```julia
a = Affine(10, 20)
```

This is easy to implement using the usual Julia syntax for constructors:

```julia
Affine(in::Integer, out::Integer) = Affine(randn(in, out), randn(1, out))
```

In practice, these constructors tend to take the parameter initialisation function as an argument so that it's more easily customisable, and use `Flux.initn` by default (which is equivalent to `randn()/100`). So `Affine`'s constructor really looks like this:

```julia
Affine(in::Integer, out::Integer; init = initn) =
  Affine(init(in, out), init(1, out))
```

## Supported syntax

The syntax used to define a forward pass like `x -> W*x + b` behaves exactly like Julia code for the most part. However, it's important to remember that it's defining a dataflow graph, not a general Julia expression. In practice this means that anything side-effectful, or things like control flow and `println`s, won't work as expected. In future we'll continue expand support for Julia syntax and features.
