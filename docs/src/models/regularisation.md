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
2-element Array{Any,1}:
 param([0.355408 0.533092; … 0.430459 0.171498])
 param([0.0, 0.0, 0.0, 0.0, 0.0])

julia> sum(norm, params(m))
26.01749952921026
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
julia> using Flux: activations

julia> c = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
Chain(Dense(10, 5, σ), Dense(5, 2), softmax)

julia> activations(c, rand(10))
3-element Array{Any,1}:
 Float32[0.84682214, 0.6704139, 0.42177814, 0.257832, 0.36255655]
 Float32[0.1501253, 0.073269576]                                 
 Float32[0.5192045, 0.48079553]                                  

julia> sum(norm, ans)
2.1166067f0
```

```@docs
Flux.activations
```
