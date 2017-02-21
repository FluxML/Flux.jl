# Флукс

[![Join the chat at https://gitter.im/MikeInnes/Flux.jl](https://badges.gitter.im/MikeInnes/Flux.jl.svg)](https://gitter.im/MikeInnes/Flux.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Build Status](https://travis-ci.org/MikeInnes/Flux.jl.svg?branch=master)](https://travis-ci.org/MikeInnes/Flux.jl)

Flux is a high-level API for machine learning, implemented in Julia.

Flux aims to provide a concise and expressive syntax for architectures that are hard to express within other frameworks. The notation should be familiar and extremely close to what you'd find in a paper or description of the model.

The current focus is on ANNs with TensorFlow or MXNet as a backend. While it's in a very early working-prototype stage, you can see what works so far in the [examples folder](/examples).

## Brief Examples

Simple multi-layer-perceptron for MNIST:

```julia
Chain(
  Input(784),
  Affine(128), relu,
  Affine( 64), relu,
  Affine( 10), softmax)
```

LSTM example:

```julia
@net type LSTM
  Wxf; Wyf; bf
  Wxi; Wyi; bi
  Wxo; Wyo; bo
  Wxc; Wyc; bc
  y; state
  function (x)
    # Gates
    forget = σ( x * Wxf + y{-1} * Wyf + bf )
    input  = σ( x * Wxi + y{-1} * Wyi + bi )
    output = σ( x * Wxo + y{-1} * Wyo + bo )
    # State update and output
    state′ = tanh( x * Wxc + y{-1} * Wyc + bc )
    state  = forget .* state{-1} + input .* state′
    y = output .* tanh(state)
  end
end

Chain(
  Input(N),
  LSTM(N, 256),
  LSTM(256, 256),
  Affine(256, N),
  softmax)
```
