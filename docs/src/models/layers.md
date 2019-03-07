## Basic Layers

These core layers form the foundation of almost all neural networks.

```@docs
Chain
Dense
```

## Convolution and Pooling Layers

These layers are used to build convolutional neural networks (CNNs).

```@docs
Conv
MaxPool
MeanPool
DepthwiseConv
ConvTranspose
```

## Recurrent Layers

Much like the core layers above, but can be used to process sequence data (as well as other kinds of structured data).

```@docs
RNN
LSTM
GRU
Flux.Recur
```

## Other General Purpose Layers
These are marginally more obscure than the Basic Layers.
But incontrast to the layers described in the other sections are not readily grouped around a paparticular purpose (e.g. CNNs or RNNs).

```@docs
Maxout
```

# Normalisation & Regularisation

These layers don't affect the structure of the network but may improve training times or reduce overfitting.

```@docs
Flux.testmode!
BatchNorm
Dropout
LayerNorm
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

## Normalisation & Regularisation

These layers don't affect the structure of the network but may improve training times or reduce overfitting.

```@docs
Flux.testmode!
BatchNorm
Dropout
AlphaDropout
LayerNorm
```
