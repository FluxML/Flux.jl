# Utility Functions

Flux contains some utility functions for working with data; these functions
help create inputs for your models or batch your dataset.
Other functions can be used to initialize your layers or to regularly execute
callback functions.

## Working with Data

```@docs
Flux.unsqueeze
Flux.stack
Flux.unstack
Flux.chunk
Flux.frequencies
Flux.batch
Flux.batchseq
Base.rpad(v::AbstractVector, n::Integer, p)
```

## Layer Initialization

These are primarily useful if you are planning to write your own layers.
Flux initializes convolutional layers and recurrent cells with `glorot_uniform`
by default.
To change the default on an applicable layer, pass the desired function with the
`init` keyword. For example:

```jldoctest; setup = :(using Flux)
julia> conv = Conv((3, 3), 1 => 8, relu; init=Flux.glorot_normal)
Conv((3, 3), 1=>8, relu)
```

```@docs
Flux.glorot_uniform
Flux.glorot_normal
Flux.kaiming_uniform
Flux.kaiming_normal
Flux.sparse_init
```

## Model Building

Flux provides some utility functions to help you generate models in an automated fashion.

[`outputsize`](@ref) enables you to calculate the output sizes of layers like [`Conv`](@ref)
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
  return Chain(conv_layers..., Dense(prod(conv_outsize), nclasses))
end
```

```@meta
DocTestFilters = [r"@ Flux .+\n"]
```
```@docs
Flux.outputsize
```
```@meta
DocTestFilters = nothing
```

## Model Abstraction

```@docs
Flux.destructure
Flux.nfan
```

## Callback Helpers

```@docs
Flux.throttle
Flux.stop
Flux.skip
```
