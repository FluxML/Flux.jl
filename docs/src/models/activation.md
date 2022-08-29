
# Activation Functions from NNlib.jl

These non-linearities used between layers of your model are exported by the [NNlib](https://github.com/FluxML/NNlib.jl) package.

Note that, unless otherwise stated, activation functions operate on scalars. To apply them to an array you can call `σ.(xs)`, `relu.(xs)` and so on. Alternatively, they can be passed to a layer like `Dense(784 => 1024, relu)` which will handle this broadcasting.

```@docs
celu
elu
gelu
hardsigmoid
sigmoid_fast
hardtanh
tanh_fast
leakyrelu
lisht
logcosh
logsigmoid
mish
relu
relu6
rrelu
selu
sigmoid
softplus
softshrink
softsign
swish
hardswish
tanhshrink
trelu
```

Julia's `Base.Math` also provide `tanh`, which can be used as an activation function:

```@docs
tanh
```