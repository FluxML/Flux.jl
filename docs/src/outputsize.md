# Size Propagation

Flux provides some utility functions to help you generate models in an automated fashion.

[`Flux.outputsize`](@ref) enables you to calculate the output sizes of layers like [`Conv`](@ref)
when applied to input samples of a given size. This is achieved by passing a "dummy" array into
the model that preserves size information without running any computation.
`outputsize(f, inputsize)` works for all layers (including custom layers) out of the box.
By default, `inputsize` expects the batch dimension,
but you can exclude the batch size with `outputsize(f, inputsize; padbatch=true)` (assuming it to be one).

Using this utility function lets you automate model building for various inputs like so:
```julia
"""
    make_model(width, height, inchannels, nclasses;
               layer_config = [16, 16, 32, 32, 64, 64])

Create a CNN for a given set of configuration parameters.

# Arguments
- `width`: the input image width
- `height`: the input image height
- `inchannels`: the number of channels in the input image
- `nclasses`: the number of output classes
- `layer_config`: a vector of the number of filters per each conv layer
"""
function make_model(width, height, inchannels, nclasses;
                    layer_config = [16, 16, 32, 32, 64, 64])
  # construct a vector of conv layers programmatically
  conv_layers = [Conv((3, 3), inchannels => layer_config[1])]
  for (infilters, outfilters) in zip(layer_config, layer_config[2:end])
    push!(conv_layers, Conv((3, 3), infilters => outfilters))
  end

  # compute the output dimensions for the conv layers
  # use padbatch=true to set the batch dimension to 1
  conv_outsize = Flux.outputsize(conv_layers, (width, height, nchannels); padbatch=true)

  # the input dimension to Dense is programatically calculated from
  #  width, height, and nchannels
  return Chain(conv_layers..., Dense(prod(conv_outsize) => nclasses))
end
```

```@docs
Flux.outputsize
```