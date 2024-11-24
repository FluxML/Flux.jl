# [How Flux Works: Gradients and Layers](@id man-basics)

## [Taking Gradients](@id man-taking-gradients)

Flux's core feature is taking gradients of Julia code. The `gradient` function takes another Julia function `f` and a set of arguments, and returns the gradient with respect to each argument. (It's a good idea to try pasting these examples in the Julia terminal.)

```jldoctest basics
julia> using Flux

julia> f(x) = 3x^2 + 2x + 1;

julia> df(x) = gradient(f, x)[1]; # df/dx = 6x + 2

julia> df(2)
14.0

julia> d2f(x) = gradient(df, x)[1]; # d²f/dx² = 6

julia> d2f(2)
6.0
```

When a function has many parameters, we can get gradients of each one at the same time:

```jldoctest basics
julia> f(x, y) = sum((x .- y).^2);

julia> gradient(f, [2, 1], [2, 0])
([0.0, 2.0], [-0.0, -2.0])
```

These gradients are based on `x` and `y`. Flux works by instead taking gradients based on the weights and biases that make up the parameters of a model.

Machine learning often can have *hundreds* of parameter arrays.
Instead of passing them to `gradient` individually, we can store them together in a structure.
The simplest example is a named tuple, created by the following syntax:

```jldoctest basics
julia> nt = (a = [2, 1], b = [2, 0], c = tanh);

julia> g(x::NamedTuple) = sum(abs2, x.a .- x.b);

julia> g(nt)
1

julia> dg_nt = gradient(g, nt)[1]
(a = [0.0, 2.0], b = [-0.0, -2.0], c = nothing)
```

Notice that `gradient` has returned a matching structure. The field `dg_nt.a` is the gradient
for `nt.a`, and so on. Some fields have no gradient, indicated by `nothing`. 

Rather than define a function like `g` every time (and think up a name for it),
it is often useful to use anonymous functions: this one is `x -> sum(abs2, x.a .- x.b)`.
Anonymous functions can be defined either with `->` or with `do`,
and such `do` blocks are often useful if you have a few steps to perform:

```jldoctest basics
julia> gradient((x, y) -> sum(abs2, x.a ./ y .- x.b), nt, [1, 2])
((a = [0.0, 0.5], b = [-0.0, -1.0], c = nothing), [-0.0, -0.25])

julia> gradient(nt, [1, 2]) do x, y
         z = x.a ./ y
         sum(abs2, z .- x.b)
       end
((a = [0.0, 0.5], b = [-0.0, -1.0], c = nothing), [-0.0, -0.25])
```

Sometimes you may want to know the value of the function, as well as its gradient.
Rather than calling the function a second time, you can call [`withgradient`](@ref Zygote.withgradient) instead:

```
julia> Flux.withgradient(g, nt)
(val = 1, grad = ((a = [0.0, 2.0], b = [-0.0, -2.0], c = nothing),))
```

## Building Simple Models

Consider a simple linear regression, which tries to predict an output array `y` from an input `x`.

```julia
predict(W, b, x) = W*x .+ b

function loss(W, b, x, y)
  ŷ = predict(W, b, x)
  sum((y .- ŷ).^2)
end

x, y = rand(5), rand(2) # Dummy data
W = rand(2, 5)
b = rand(2)

loss(W, b, x, y) # ~ 3
```

To improve the prediction we can take the gradients of the loss with respect to `W` and `b` and perform gradient descent.

```julia
using Flux

dW, db = gradient((W, b) -> loss(W, b, x, y), W, b)
```

Now that we have gradients, we can pull them out and update `W` to train the model.

```julia
W .-= 0.1 .* dW

loss(W, b, x, y) # ~ 2.5
```

The loss has decreased a little, meaning that our prediction `x` is closer to the target `y`. If we have some data we can already try [training the model](../training/training.md).

All deep learning in Flux, however complex, is a simple generalisation of this example. Of course, models can *look* very different – they might have millions of parameters or complex control flow. Let's see how Flux handles more complex models.

## Building Layers

It's common to create more complex models than the linear regression above. For example, we might want to have two linear layers with a nonlinearity like [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) in between them. We could write this as:

```julia
using Flux

W1 = rand(3, 5)
b1 = rand(3)
layer1(x) = W1 * x .+ b1

W2 = rand(2, 3)
b2 = rand(2)
layer2(x) = W2 * x .+ b2

model(x) = layer2(sigmoid.(layer1(x)))

model(rand(5)) # => 2-element vector
```

This works but is fairly unwieldy, with a lot of repetition – especially as we add more layers. One way to factor this out is to create a function that returns linear layers.

```julia
function linear(in, out)
  W = randn(out, in)
  b = randn(out)
  x -> W * x .+ b
end

linear1 = linear(5, 3) # we can access linear1.W etc
linear2 = linear(3, 2)

model(x) = linear2(sigmoid.(linear1(x)))

model(rand(5)) # => 2-element vector
```

Another (equivalent) way is to create a struct that explicitly represents the affine layer.

```julia
struct Affine
  W
  b
end

Affine(in::Integer, out::Integer) =
  Affine(randn(out, in), zeros(out))

# Overload call, so the object can be used as a function
(m::Affine)(x) = m.W * x .+ m.b

a = Affine(10, 5)

a(rand(10)) # => 5-element vector
```

Congratulations! You just built the [`Dense`](@ref) layer that comes with Flux. Flux has many interesting layers available, but they're all things you could have built yourself very easily.

(There is one small difference with `Dense` – for convenience it also takes an activation function, like `Dense(10 => 5, sigmoid)`.)

## Stacking It Up

It's pretty common to write models that look something like:

```julia
layer1 = Dense(10 => 5, relu)
# ...
model(x) = layer3(layer2(layer1(x)))
```

For long chains, it might be a bit more intuitive to have a list of layers, like this:

```julia
using Flux

layers = [Dense(10 => 5, relu), Dense(5 => 2), softmax]

model(x) = foldl((x, m) -> m(x), layers, init = x)

model(rand(10)) # => 2-element vector
```

Handily, this is also provided for in Flux:

```julia
model2 = Chain(
  Dense(10 => 5, relu),
  Dense(5 => 2),
  softmax)

model2(rand(10)) # => 2-element vector
```

This quickly starts to look like a high-level deep learning library; yet you can see how it falls out of simple abstractions, and we lose none of the power of Julia code.

A nice property of this approach is that because "models" are just functions (possibly with trainable parameters), you can also see this as simple function composition.

```julia
m = Dense(5 => 2) ∘ Dense(10 => 5, σ)

m(rand(10))
```

Likewise, `Chain` will happily work with any Julia function.

```julia
m = Chain(x -> x^2, x -> x+1)

m(5) # => 26
```

## Layer Helpers

We can give our layer some additional functionality, like nice printing, using the [`@layer`](@ref Flux.@layer) macro:

```julia
Flux.@layer Affine
```

Finally, most Flux layers make bias optional, and allow you to supply the function used for generating random weights. We can easily add these refinements to the `Affine` layer as follows, using the helper function [`create_bias`](@ref Flux.create_bias):

```julia
function Affine((in, out)::Pair; bias=true, init=glorot_uniform)
  W = init(out, in)
  b = Flux.create_bias(W, bias, out)
  return Affine(W, b)
end

Affine(3 => 1, bias=false) |> gpu
```

