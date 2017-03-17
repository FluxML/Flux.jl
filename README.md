# Флукс

[![Build Status](https://travis-ci.org/MikeInnes/Flux.jl.svg?branch=master)](https://travis-ci.org/MikeInnes/Flux.jl) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://mikeinnes.github.io/Flux.jl/stable) [![Join the chat at https://gitter.im/MikeInnes/Flux.jl](https://badges.gitter.im/MikeInnes/Flux.jl.svg)](https://gitter.im/MikeInnes/Flux.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Flux is a high-level library for machine learning, implemented in Julia.

Flux is designed to get the best performance (by running on TensorFlow or MXNet) while still being intuitive to work with – you get good error messages, can step through models with the debugger, and the notation is very close to what you'd find in a paper.

Check out the [docs](https://mikeinnes.github.io/Flux.jl/latest/) to get started. Flux is in alpha so **please open issues liberally**; if something is broken for you it can most likely be fixed easily, or if you're not sure how to do something we can help.

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
