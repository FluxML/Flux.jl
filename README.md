# Флукс

[![Build Status](https://travis-ci.org/FluxML/Flux.jl.svg?branch=master)](https://travis-ci.org/FluxML/Flux.jl) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://fluxml.github.io/Flux.jl/stable/) [![Join the chat at https://gitter.im/FluxML](https://badges.gitter.im/FluxML/Lobby.svg)](https://gitter.im/FluxML/Lobby) [Slack](https://discourse.julialang.org/t/announcing-a-julia-slack/4866)

Flux is an unusually elegant machine learning library. It provides lightweight abstractions on top of Julia's native GPU and AD support, while remaining fully hackable (right down to the [GPU kernels](https://github.com/FluxML/CuArrays.jl)).

Define a simple model using any Julia code:

```julia
using Flux.Tracker
x, y = rand(10), rand(5) # Dummy input / output
# `track` defines parameters that we can train
W, b = track(randn(5,10)), track(randn(5))
# Transform `x` and calculate the mean squared error
loss = Flux.mse(W*x .+ b, y)
# Calculate and store gradients of `track`ed parameters
back!(loss)
Tracker.grad(W) # Get the gradient of `W` wrt the loss
```

Define a larger model using high-level abstractions:

```julia
using Flux

m = Chain(
  Dense(10, 32, relu),
  Dense(32, 10), softmax)

m(rand(10))
```

Mix and match the two:

```julia
using Flux.Tracker
x, y = rand(10), rand(5)
d = Dense(10, 5)
loss = Flux.mse(d(x), y)
```

See the [documentation](http://fluxml.github.io/Flux.jl/stable/) or the [model zoo](https://github.com/FluxML/model-zoo/) for more examples.
