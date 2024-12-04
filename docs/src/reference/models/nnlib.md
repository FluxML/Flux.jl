```@meta
CollapsedDocStrings = true
```

# Neural Network primitives from NNlib.jl

Flux re-exports all of the functions exported by the [NNlib](https://github.com/FluxML/NNlib.jl) package. This includes activation functions, described on [their own page](@ref man-activations). Many of the functions on this page exist primarily as the internal implementation of Flux layer, but can also be used independently.


## Attention

Primitives for the [`MultiHeadAttention`](@ref) layer.

```@docs
NNlib.dot_product_attention
NNlib.dot_product_attention_scores
NNlib.make_causal_mask
```

## Softmax

`Flux`'s [`Flux.logitcrossentropy`](@ref) uses [`NNlib.logsoftmax`](@ref) internally.

```@docs
softmax
logsoftmax
```

## Pooling

`Flux`'s [`AdaptiveMaxPool`](@ref), [`AdaptiveMeanPool`](@ref), [`GlobalMaxPool`](@ref), [`GlobalMeanPool`](@ref), 
[`MaxPool`](@ref), and [`MeanPool`](@ref) use [`NNlib.PoolDims`](@ref), [`NNlib.maxpool`](@ref), and [`NNlib.meanpool`](@ref) as their backend.

```@docs
NNlib.PoolDims
NNlib.lpnormpool
NNlib.maxpool
NNlib.meanpool
```

## Padding

```@docs
NNlib.pad_circular
NNlib.pad_constant
NNlib.pad_reflect
NNlib.pad_repeat
NNlib.pad_symmetric
NNlib.pad_zeros
```

## Convolution

`Flux`'s [`Conv`](@ref) and [`CrossCor`](@ref) layers use [`NNlib.DenseConvDims`](@ref) and [`NNlib.conv`](@ref) internally. 

```@docs
conv
ConvDims
depthwiseconv
DepthwiseConvDims
DenseConvDims
```

## Dropout

```@docs
NNlib.dropout
NNlib.dropout!
```

## Upsampling

`Flux`'s [`Upsample`](@ref) layer uses [`NNlib.upsample_nearest`](@ref), [`NNlib.upsample_bilinear`](@ref), and [`NNlib.upsample_trilinear`](@ref) as its backend. Additionally, `Flux`'s [`PixelShuffle`](@ref) layer uses [`NNlib.pixel_shuffle`](@ref) as its backend.

```@docs
upsample_nearest
upsample_linear
∇upsample_linear
upsample_bilinear
∇upsample_bilinear
upsample_trilinear
∇upsample_trilinear
pixel_shuffle
```

## Batched Operations

`Flux`'s [`Flux.Bilinear`](@ref) layer uses [`NNlib.batched_mul`](@ref) internally.

```@docs
batched_mul
batched_mul!
batched_adjoint
batched_transpose
batched_vec
```

## Gather and Scatter

`Flux`'s [`Embedding`](@ref) layer uses [`NNlib.gather`](@ref) as its backend.

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
