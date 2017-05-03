# Model Building Basics

## Net Functions

Flux's core feature is the `@net` macro, which adds some superpowers to regular ol' Julia functions. Consider this simple function with the `@net` annotation applied:

```julia
@net f(x) = x .* x
f([1,2,3]) == [1,4,9]
```

This behaves as expected, but we have some extra features. For example, we can convert the function to run on [TensorFlow](https://www.tensorflow.org/) or  [MXNet](https://github.com/dmlc/MXNet.jl):

```julia
f_mxnet = mxnet(f)
f_mxnet([1,2,3]) == [1.0, 4.0, 9.0]
```

Simples! Flux took care of a lot of boilerplate for us and just ran the multiplication on MXNet. MXNet can optimise this code for us, taking advantage of parallelism or running the code on a GPU.

Using MXNet, we can get the gradient of the function, too:

```julia
back!(f_mxnet, [1,1,1], [1,2,3]) == ([2.0, 4.0, 6.0])
```

`f` is effectively `x^2`, so the gradient is `2x` as expected.

For TensorFlow users this may seem similar to building a graph as usual. The difference is that Julia code still behaves like Julia code. Error messages give you helpful stacktraces that pinpoint mistakes. You can step through the code in the debugger. The code runs when it's called, as usual, rather than running once to build the graph and then again to execute it.

## The Model

The core concept in Flux is the *model*. This corresponds to what might be called a "layer" or "module" in other frameworks. A model is simply a differentiable function with parameters. Given a model `m` we can do things like:

```julia
m(x)           # See what the model does to an input vector `x`
back!(m, Δ, x) # backpropogate the gradient `Δ` through `m`
update!(m, η)  # update the parameters of `m` using the gradient
```

We can implement a model however we like as long as it fits this interface. But as hinted above, `@net` is a particularly easy way to do it, as `@net` functions are models already.

## Parameters

Consider how we'd write a logistic regression. We just take the Julia code and add `@net`.

```julia
W = randn(3,5)
b = randn(3)
@net logistic(x) = softmax(W * x + b)

x1 = rand(5) # [0.581466,0.606507,0.981732,0.488618,0.415414]
y1 = logistic(x1) # [0.32676,0.0974173,0.575823]
```

<!-- TODO -->

## Layers

Bigger networks contain many affine transformations like `W * x + b`. We don't want to write out the definition every time we use it. Instead, we can factor this out by making a function that produces models:

```julia
function create_affine(in, out)
  W = randn(out,in)
  b = randn(out)
  @net x -> W * x + b
end

affine1 = create_affine(3,2)
affine1([1,2,3])
```

Flux has a [more powerful syntax](templates.html) for this pattern, but also provides a bunch of layers out of the box. So we can instead write:

```julia
affine1 = Affine(5, 5)
affine2 = Affine(5, 5)

softmax(affine1(x1)) # [0.167952, 0.186325, 0.176683, 0.238571, 0.23047]
softmax(affine2(x1)) # [0.125361, 0.246448, 0.21966, 0.124596, 0.283935]
```

## Combining Layers

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

## Dressed like a model

We noted above that a model is a function with trainable parameters. Normal functions like `exp` are actually models too, that happen to have 0 parameters. Flux doesn't care, and anywhere that you use one, you can use the other. For example, `Chain` will happily work with regular functions:

```julia
foo = Chain(exp, sum, log)
foo([1,2,3]) == 3.408 == log(sum(exp([1,2,3])))
```
