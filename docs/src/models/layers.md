## Basic Layers

These core layers form the foundation of almost all neural networks.

```@docs
Chain
Dense
```

## Recurrent Layers

Much like the core layers above, but can be used to process sequence data (as well as other kinds of structured data).

```@docs
RNN
LSTM
Flux.Recur
```

## Activation Functions

Non-linearities that go between layers of your model. Most of these functions are defined in [NNlib](https://github.com/FluxML/NNlib.jl) but are available by default in Flux.

Note that, unless otherwise stated, activation functions operate on scalars. To apply them to an array you can call `σ.(xs)`, `relu.(xs)` and so on.

```@docs
σ
relu
leakyrelu
elu
swish
```
