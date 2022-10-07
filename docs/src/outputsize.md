# Shape Inference

Flux has some tools to help generate models in an automated fashion, by inferring the size
of arrays that layers will recieve, without doing any computation. 
This is especially useful for convolutional models, where the same [`Conv`](@ref) layer
accepts any size of image, but the next layer may not. 

The higher-level tool is a macro [`@autosize`](@ref) which acts on the code defining the layers,
and replaces each appearance of `_` with the relevant size. This simple example returns a model
with `Dense(845 => 10)` as the last layer:

```julia
@autosize (28, 28, 1, 32) Chain(Conv((3, 3), _ => 5, relu, stride=2), Flux.flatten, Dense(_ => 10))
```

The input size may be provided at runtime, like `@autosize (sz..., 1, 32) Chain(Conv(`..., but all the
layer constructors containing `_` must be explicitly written out -- the macro sees the code as written.

This macro relies on a lower-level function [`outputsize`](@ref Flux.outputsize), which you can also use directly:

```julia
c = Conv((3, 3), 1 => 5, relu, stride=2)
Flux.outputsize(c, (28, 28, 1, 32))  # returns (13, 13, 5, 32)
```

The function `outputsize` works by passing a "dummy" array into the model, which propagates through very cheaply.
It should work for all layers, including custom layers, out of the box.

An example of how to automate model building is this:
```julia
"""
    make_model(width, height, [inchannels, nclasses; layer_config])

Create a CNN for a given set of configuration parameters. Arguments:
- `width`, `height`: the input image size in pixels
- `inchannels`: the number of channels in the input image, default `1`
- `nclasses`: the number of output classes, default `10`
- Keyword `layer_config`: a vector of the number of filters per layer, default `[16, 16, 32, 64]`
"""
function make_model(width, height, inchannels = 1, nclasses = 10;
                    layer_config = [16, 16, 32, 64])
  # construct a vector of layers:
  conv_layers = []
  push!(conv_layers, Conv((5, 5), inchannels => layer_config[1], relu, pad=SamePad()))
  for (inch, outch) in zip(layer_config, layer_config[2:end])
    push!(conv_layers, Conv((3, 3), inch => outch, sigmoid, stride=2))
  end

  # compute the output dimensions after these conv layers:
  conv_outsize = Flux.outputsize(conv_layers, (width, height, inchannels); padbatch=true)

  # use this to define appropriate Dense layer:
  last_layer = Dense(prod(conv_outsize) => nclasses)
  return Chain(conv_layers..., Flux.flatten, last_layer)
end

m = make_model(28, 28, 3, layer_config = [9, 17, 33, 65])

Flux.outputsize(m, (28, 28, 3, 42)) == (10, 42) == size(m(randn(Float32, 28, 28, 3, 42)))
```

Alternatively, using the macro, the definition of `make_model` could end with:

```
  # compute the output dimensions & construct appropriate Dense layer:
  return @autosize (width, height, inchannels, 1) Chain(conv_layers..., Flux.flatten, Dense(_ => nclasses))
end
```

### Listing

```@docs
Flux.@autosize
Flux.outputsize
```
