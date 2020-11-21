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
```

## Model Building

Flux provides some utility functions to help you generate models in an automated fashion.

[`outdims`](@ref) enables you to calculate the spatial output dimensions of layers like [`Conv`](@ref) when applied to input images of a given size. This is achieved by passing a "dummy" array into the model that preserves size information without running any computation. `outdims(f, isize)` works for all layers (including custom layers) out of the box as long as `isize` includes the batch dimension. If [`Flux.dimhint`](@ref) is defined for a layer, then `isize` may drop the batch dimension.

Using this utility function lets you automate model building for various inputs like so:
```julia
function make_model(width, height, nchannels, nclasses)
  # returns 1D array (vector) of conv layers
  conv_layers = make_conv(width, height, nchannels)
  conv_outsize = outdims(conv_layers, (width, height, nchannels))

  # the input dimension to Dense is programatically calculated from
  #  width, height, and nchannels
  return Chain(conv_layers..., Dense(prod(conv_outsize), nclasses))
end
```

```@docs
Flux.outdims
Flux.dimhint
```

## Model Abstraction

```@docs
Flux.destructure
```

## Callback Helpers

```@docs
Flux.throttle
Flux.stop
Flux.skip
```
