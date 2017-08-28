# Model Templates

We mentioned that we could factor out the repetition of defining affine layers with something like:

```julia
function create_affine(in, out)
  W = param(randn(out,in))
  b = param(randn(out))
  @net x -> W * x + b
end
```

`@net type` syntax provides a shortcut for this:

```julia
@net mutable struct MyAffine
  W
  b
  x -> x * W + b
end

# Convenience constructor
MyAffine(in::Integer, out::Integer) =
  MyAffine(randn(out, in), randn(out))

model = Chain(MyAffine(5, 5), MyAffine(5, 5))

model(x1) # [-1.54458,0.492025,0.88687,1.93834,-4.70062]
```

This is almost exactly how `Affine` is defined in Flux itself. Using `@net type` gives us some extra conveniences:

* It creates default constructor `MyAffine(::AbstractArray, ::AbstractArray)` which initialises `param`s for us;
* It subtypes `Flux.Model` to explicitly mark this as a model;
* We can easily define custom constructors or instantiate `Affine` with arbitrary weights of our choosing;
* We can dispatch on the `Affine` type, for example to override how it gets converted to MXNet, or to hook into shape inference.

## Models in templates

`@net` models can contain sub-models as well as just array parameters:

```julia
@net mutable struct TLP
  first
  second
  function (x)
    l1 = σ(first(x))
    l2 = softmax(second(l1))
  end
end
```

Clearly, the `first` and `second` parameters are not arrays here, but should be models themselves, and produce a result when called with an input array `x`. The `Affine` layer fits the bill, so we can instantiate `TLP` with two of them:

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

## Supported syntax

The syntax used to define a forward pass like `x -> x*W + b` behaves exactly like Julia code for the most part. However, it's important to remember that it's defining a dataflow graph, not a general Julia expression. In practice this means that anything side-effectful, or things like control flow and `println`s, won't work as expected. In future we'll continue to expand support for Julia syntax and features.
