# Additional Convolution Layers

## Depthwise Convolutions

Using Depthwise Convolutions is pretty straightforword and much similar
to the usage of normal Convolutions. So simply we can swap in a
Depthwise Convolution in place of a Convolution.

Lets say we have to define a simple convolution layer like
```julia
m = Conv((3, 3), 3=>64, pad = (1, 1))
```

The alternative to this using a Depthwise Convolution would be
```julia
m = Chain(DepthwiseConv((3, 3), 3=>2, pad = (1, 1)),
          Conv((1, 1), 6=>64))
```

Incase the second argument to `DepthwiseConv` is an `Integer` instead of a
`Pair` the channel multiplier is taken to be 1.
