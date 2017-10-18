## Model Layers

These core layers form the foundation of almost all neural networks.

```@docs
Chain
Dense
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
softmax
```
