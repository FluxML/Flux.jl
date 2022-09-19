# Neural Network primitives from NNlib.jl

Flux re-exports all of the functions exported by the [NNlib](https://github.com/FluxML/NNlib.jl) package. This includes activation functions, described on the next page. Many of the functions on this page exist primarily as the internal implementation of Flux layer, but can also be used independently.

## Softmax

`Flux`'s `logitcrossentropy` uses `NNlib.softmax` internally.

```@docs
softmax
logsoftmax
```

## Pooling

`Flux`'s `AdaptiveMaxPool`, `AdaptiveMeanPool`, `GlobalMaxPool`, `GlobalMeanPool`, `MaxPool`, and `MeanPool` use `NNlib.PoolDims`, `NNlib.maxpool`, and `NNlib.meanpool` as their backend.

```@docs
PoolDims
maxpool
meanpool
```

## Padding

```@docs
pad_reflect
pad_constant
pad_repeat
pad_zeros
```

## Convolution

`Flux`'s `Conv` and `CrossCor` layers use `NNlib.DenseConvDims` and `NNlib.conv` internally. 

```@docs
conv
ConvDims
depthwiseconv
DepthwiseConvDims
DenseConvDims
```

## Upsampling

`Flux`'s `Upsample` layer uses `NNlib.upsample_nearest`, `NNlib.upsample_bilinear`, and `NNlib.upsample_trilinear` as its backend. Additionally, `Flux`'s `PixelShuffle` layer uses `NNlib.pixel_shuffle` as its backend.

```@docs
upsample_nearest
∇upsample_nearest
upsample_linear
∇upsample_linear
upsample_bilinear
∇upsample_bilinear
upsample_trilinear
∇upsample_trilinear
pixel_shuffle
```

## Batched Operations

`Flux`'s `Bilinear` layer uses `NNlib.batched_mul` internally.

```@docs
batched_mul
batched_mul!
batched_adjoint
batched_transpose
batched_vec
```

## Gather and Scatter

`Flux`'s `Embedding` layer uses `NNlib.gather` as its backend.

```@docs
NNlib.gather
NNlib.gather!
NNlib.scatter
NNlib.scatter!
```

## Sampling

```@docs
grid_sample
∇grid_sample
```

## Losses

```@docs
ctc_loss
```

## Miscellaneous

```@docs
logsumexp
NNlib.glu
```
