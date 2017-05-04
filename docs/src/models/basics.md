# Model Building Basics

## Net Functions

Flux's core feature is the `@net` macro, which adds some superpowers to regular ol' Julia functions. Consider this simple function with the `@net` annotation applied:

```julia
@net f(x) = x .* x
f([1,2,3]) == [1,4,9]
```

This behaves as expected, but we have some extra features. For example, we can convert the function to run on [TensorFlow](https://www.tensorflow.org/) or [MXNet](https://github.com/dmlc/MXNet.jl):

```julia
f_mxnet = mxnet(f)
f_mxnet([1,2,3]) == [1.0, 4.0, 9.0]
```

Simples! Flux took care of a lot of boilerplate for us and just ran the multiplication on MXNet. MXNet can optimise this code for us, taking advantage of parallelism or running the code on a GPU.

Using MXNet, we can get the gradient of the function, too:

```julia
back!(f_mxnet, [1,1,1], [1,2,3]) == ([2.0, 4.0, 6.0],)
```

`f` is effectively `x^2`, so the gradient is `2x` as expected.

## The Model

The core concept in Flux is the *model*. This corresponds to what might be called a "layer" or "module" in other frameworks. A model is simply a differentiable function with parameters. Given a model `m` we can do things like:

```julia
m(x)           # See what the model does to an input vector `x`
back!(m, Δ, x) # backpropogate the gradient `Δ` through `m`
update!(m, η)  # update the parameters of `m` using the gradient
```

We can implement a model however we like as long as it fits this interface. But as hinted above, `@net` is a particularly easy way to do it, because it gives you these functions for free.

## Parameters

Consider how we'd write a logistic regression. We just take the Julia code and add `@net`.

```julia
@net logistic(W, b, x) = softmax(x * W .+ b)

W = randn(10, 2)
b = randn(1, 2)
x = rand(1, 10) # [0.563 0.346 0.780  …] – fake data
y = [1 0] # our desired classification of `x`

ŷ = logistic(W, b, x) # [0.46 0.54]
```

The network takes a set of 10 features (`x`, a row vector) and produces a classification `ŷ`, equivalent to a probability of true vs false. `softmax` scales the output to sum to one, so that we can interpret it as a probability distribution.

We can use MXNet and get gradients:

```julia
logisticm = mxnet(logistic)
logisticm(W, b, x) # [0.46 0.54]
back!(logisticm, [0.1 -0.1], W, b, x) # (dW, db, dx)
```

The gradient `[0.1 -0.1]` says that we want to increase `ŷ[1]` and decrease `ŷ[2]` to get closer to `y`. `back!` gives us the tweaks we need to make to each input (`W`, `b`, `x`) in order to do this. If we add these tweaks to `W` and `b` it will predict `ŷ` more accurately.

Treating parameters like `W` and `b` as inputs can get unwieldy in larger networks. Since they are both global we can use them directly:

```julia
@net logistic(x) = softmax(x * W .+ b)
```

However, this gives us a problem: how do we get their gradients?

Flux solves this with the `Param` wrapper:

```julia
W = param(randn(10, 2))
b = param(randn(1, 2))
@net logistic(x) = softmax(x * W .+ b)
```

This works as before, but now `W.x` stores the real value and `W.Δx` stores its gradient, so we don't have to manage it by hand. We can even use `update!` to apply the gradients automatically.

```julia
logisticm(x) # [0.46, 0.54]

back!(logisticm, [-1 1], x)
update!(logisticm, 0.1)

logisticm(x) # [0.51, 0.49]
```

Our network got a little closer to the target `y`. Now we just need to repeat this millions of times.

*Side note:* We obviously need a way to calculate the "tweak" `[0.1, -0.1]` automatically. We can use a loss function like *mean squared error* for this:

```julia
# How wrong is ŷ?
mse([0.46, 0.54], [1, 0]) == 0.292
# What change to `ŷ` will reduce the wrongness?
back!(mse, -1, [0.46, 0.54], [1, 0]) == [0.54 -0.54]
```

## Layers

Bigger networks contain many affine transformations like `W * x + b`. We don't want to write out the definition every time we use it. Instead, we can factor this out by making a function that produces models:

```julia
function create_affine(in, out)
  W = param(randn(out,in))
  b = param(randn(out))
  @net x -> W * x + b
end

affine1 = create_affine(3,2)
affine1([1,2,3])
```

Flux has a [more powerful syntax](templates.html) for this pattern, but also provides a bunch of layers out of the box. So we can instead write:

```julia
affine1 = Affine(5, 5)
affine2 = Affine(5, 5)

softmax(affine1(x)) # [0.167952 0.186325 0.176683 0.238571 0.23047]
softmax(affine2(x)) # [0.125361 0.246448 0.21966 0.124596 0.283935]
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

## Dressed like a model

We noted above that a model is a function with trainable parameters. Normal functions like `exp` are actually models too – they just happen to have 0 parameters. Flux doesn't care, and anywhere that you use one, you can use the other. For example, `Chain` will happily work with regular functions:

```julia
foo = Chain(exp, sum, log)
foo([1,2,3]) == 3.408 == log(sum(exp([1,2,3])))
```
