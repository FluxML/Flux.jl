# Regularisation

Applying regularisation to model parameters is straightforward. We just need to
apply an appropriate regulariser, such as `norm`, to each model parameter and
add the result to the overall loss.

For example, say we have a simple regression.

```julia
using Flux: crossentropy
m = Dense(10, 5)
loss(x, y) = crossentropy(softmax(m(x)), y)
```

We can regularise this by taking the (L2) norm of the parameters, `m.W` and `m.b`.

```julia
using LinearAlgebra
penalty() = norm(m.W) + norm(m.b)
loss(x, y) = crossentropy(softmax(m(x)), y) + penalty()
```

When working with layers, Flux provides the `params` function to grab all
parameters at once. We can easily penalise everything with `sum(norm, params)`.

```julia
julia> params(m)
Params([Float32[-0.0123748 -0.310727 … 0.557616 -0.365492; 0.507911 0.333276 … -0.299706 -0.350524; … ; 0.399712 -0.0647629 … 0.0437486 0.443338; 0.526206 0.121937 … -0.627679 -0.481252] (tracked), Float32[0.0, 0.0, 0.0, 0.0, 0.0] (tracked)])

julia> sum(norm, params(m))
2.626020868792272 (tracked)
```

Here's a larger example with a multi-layer perceptron.

```julia
m = Chain(
  Dense(28^2, 128, relu),
  Dense(128, 32, relu),
  Dense(32, 10), softmax)

loss(x, y) = crossentropy(m(x), y) + sum(norm, params(m))

loss(rand(28^2), rand(10))
```

One can also easily add per-layer regularisation via the `activations` function:

```julia
julia> c = Chain(Dense(10,5,σ),Dense(5,2),softmax)
Chain(Dense(10, 5, NNlib.σ), Dense(5, 2), NNlib.softmax)

julia> activations(c, rand(10))
3-element Array{Any,1}:
 param([0.71068, 0.831145, 0.751219, 0.227116, 0.553074])
 param([0.0330606, -0.456104])
 param([0.61991, 0.38009])

julia> sum(norm, ans)
2.639678767773633 (tracked)
```
