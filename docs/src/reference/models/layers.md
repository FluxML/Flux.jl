```@meta
CurrentModule = Flux
CollapsedDocStrings = true
```

# Built-in Layer Types

If you started at the beginning of the guide, then you have already met the
basic [`Dense`](@ref) layer, and seen [`Chain`](@ref) for combining layers.
These core layers form the foundation of almost all neural networks.

The `Dense` exemplifies several features:

* It contains an an [activation function](@ref man-activations), which is broadcasted over the output. Because this broadcast can be fused with other operations, doing so is more efficient than applying the activation function separately.

* It take an `init` keyword, which accepts a function acting like `rand`. That is, `init(2,3,4)` should create an array of this size. Flux has [many such functions](@ref man-init-funcs) built-in. All make a CPU array, moved later with [`gpu`](@ref Flux.gpu) if desired.

* The bias vector is always initialised [`Flux.zeros32`](@ref). The keyword `bias=false` will turn this off, i.e. keeping the bias permanently zero.

* It is annotated with [`@layer`](@ref Flux.@layer), which means that [`Flux.setup`](@ref Flux.setup) will see the contents, and [`gpu`](@ref Flux.gpu) will move their arrays to the GPU.

By contrast, `Chain` itself contains no parameters, but connects other layers together.
The section on [dataflow layers](@ref man-dataflow-layers) introduces others like this.

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

To understand how strides and padding work, the article by [Dumoulin & Visin](https://arxiv.org/abs/1603.07285) has great illustrations.

```@docs
Conv
ConvTranspose
CrossCor
DepthwiseConv
SamePad
```

## MultiHeadAttention

The basic blocks needed to implement [Transformer](https://arxiv.org/abs/1706.03762) architectures. See also the functional counterparts
documented in NNlib's [Attention](@ref) section.

```@docs
MultiHeadAttention
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

## [Dataflow Layers, or Containers](@id man-dataflow-layers)

The basic `Chain(F, G, H)` applies the layers it contains in sequence, equivalent to `H ∘ G ∘ F`. Flux has some other layers which contain layers, but connect them up in a more complicated way: `SkipConnection` allows ResNet's residual connection.

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
RNNCell
RNN
LSTMCell
LSTM
GRUCell
GRU
GRUv3Cell
GRUv3
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
```

### Test vs. Train

Several normalisation layers behave differently under training and inference (testing). By default, Flux will automatically determine when a layer evaluation is part of training or inference. 

!!! warning
    This automatic train/test detection works best with Zygote, the default
    automatic differentiation package. It may not work with other packages
    such as Tracker, Yota, or ForwardDiff.

The functions `Flux.trainmode!` and `Flux.testmode!` let you manually specify which behaviour you want. When called on a model, they will place all layers within the model into the specified mode.

```@docs
testmode!(::Any)
testmode!(::Any, ::Any)
trainmode!
```
