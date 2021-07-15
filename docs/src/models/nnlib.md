# NNlib

Flux re-exports all of the functions exported by the [NNlib](https://github.com/FluxML/NNlib.jl) package.

## Activation Functions

Non-linearities that go between layers of your model. Note that, unless otherwise stated, activation functions operate on scalars. To apply them to an array you can call `σ.(xs)`, `relu.(xs)` and so on.

```@docs
NNlib.celu
NNlib.elu
NNlib.gelu
NNlib.hardsigmoid
NNlib.hardtanh
NNlib.leakyrelu
NNlib.lisht
NNlib.logcosh
NNlib.logsigmoid
NNlib.mish
NNlib.relu
NNlib.relu6
NNlib.rrelu
NNlib.selu
NNlib.sigmoid
NNlib.softplus
NNlib.softshrink
NNlib.softsign
NNlib.swish
NNlib.tanhshrink
NNlib.trelu
```

## Softmax

```@docs
NNlib.softmax
NNlib.logsoftmax
```

## Pooling

```@docs
NNlib.maxpool
NNlib.meanpool
```

## Convolution

```@docs
NNlib.conv
NNlib.depthwiseconv
```

## Upsampling

```@docs
NNlib.upsample_nearest
NNlib.upsample_bilinear
NNlib.pixel_shuffle
```

## Batched Operations

```@docs
NNlib.batched_mul
NNlib.batched_mul!
NNlib.batched_adjoint
NNlib.batched_transpose
```

## Gather and Scatter

```@docs
NNlib.gather
NNlib.scatter
```
