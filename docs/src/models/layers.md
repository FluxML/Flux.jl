# Built-in Layer Types

If you started at the beginning, then you have already met the basic [`Dense`](@ref) layer, and seen [`Chain`](@ref) for combining layers. These core layers form the foundation of almost all neural networks. 

The `Dense` layer 

* Weight matrices are created ... Many layers take an `init` keyword, accepts a function acting like `rand`. That is, `init(2,3,4)` creates an array of this size.  ... always on the CPU. 

* An activation function. This is broadcast over the output: `Flux.Scale(3, tanh)([1,2,3]) ≈ tanh.(1:3)`

* The bias vector is always intialised `Flux.zeros32`. The keyword `bias=false` will turn this off.


* All layers are annotated with `@layer`, which means that `params` will see the contents, and `gpu` will move their arrays to the GPU.


## Fully Connected

```@docs
Dense
Flux.Bilinear
Flux.Scale
```

Perhaps `Scale` isn't quite fully connected, but it may be thought of as `Dense(Diagonal(s.weights), s.bias)`, and LinearAlgebra's `Diagonal` is a matrix which just happens to contain many zeros.

## Convolution Models

These layers are used to build convolutional neural networks (CNNs).

They all expect images in what is called WHCN order: a batch of 32 colour images, each 50 x 50 pixels, will have `size(x) == (50, 50, 3, 32)`. A single grayscale image might instead have `size(x) == (28, 28, 1, 1)`.

Besides images, 2D data, they also work with 1D data, where for instance stereo sound recording with 1000 samples might have `size(x) == (1000, 2, 1)`. They will also work with 3D data, `ndims(x) == 5`, where again the last two dimensions are channel and batch.

To understand how `stride` ?? there's a cute article.

```@docs
Conv
Conv(weight::AbstractArray)
ConvTranspose
ConvTranspose(weight::AbstractArray)
CrossCor
CrossCor(weight::AbstractArray)
DepthwiseConv
SamePad
Flux.flatten
```

### Pooling

These layers are commonly used after a convolution layer, and reduce the size of its output. They have no trainable parameters.

```@docs
AdaptiveMaxPool
MaxPool
GlobalMaxPool
AdaptiveMeanPool
MeanPool
GlobalMeanPool
```

## Upsampling

The opposite of pooling, these layers increase the size of an array. They have no trainable parameters. 

```@docs
Upsample
PixelShuffle
```

## Embedding Vectors

These layers accept an index, and return a vector (or several indices, and several vectors). The possible embedding vectors are learned parameters.

```@docs
Flux.Embedding
Flux.EmbeddingBag
```

## Dataflow Layers, or Containers

The basic `Chain(F, G, H)` applies the layers it contains in sequence, equivalent to `H ∘ G ∘ F`. Flux has some other layers which contain layers, but connect them up in a more complicated way: `SkipConnection` allows ResNet's ??residual connection.

These are all defined with [`@layer`](@ref)` :exand TypeName`, which tells the pretty-printing code that they contain other layers.

```@docs
Chain
Flux.activations
Maxout
SkipConnection
Parallel
PairwiseFusion
```

## Recurrent Models

Much like the core layers above, but can be used to process sequence data (as well as other kinds of structured data).

```@docs
RNN
LSTM
GRU
GRUv3
Flux.Recur
Flux.reset!
```

## Normalisation & Regularisation

These layers don't affect the structure of the network but may improve training times or reduce overfitting. Some of them contain trainable parameters, while others do not.

```@docs
BatchNorm
Dropout
AlphaDropout
LayerNorm
InstanceNorm
GroupNorm
Flux.normalise
Flux.dropout
```

### Test vs. Train

Several normalisation layers behave differently under training and inference (testing). By default, Flux will automatically determine when a layer evaluation is part of training or inference. 

The functions `Flux.trainmode!` and `Flux.testmode!` let you manually specify which behaviour you want. When called on a model, they will place all layers within the model into the specified mode.

```@docs
Flux.testmode!
trainmode!
```
