# [Random Weight Initialisation](@id man-init-funcs)

Flux initialises convolutional layers and recurrent cells with `glorot_uniform` by default.
Most layers accept a function as an `init` keyword, which replaces this default. For example:

```jldoctest; setup = :(using Flux)
julia> conv = Conv((3, 3), 3 => 2, relu; init=Flux.glorot_normal)
Conv((3, 3), 3 => 2, relu)  # 56 parameters

julia> conv.bias
2-element Vector{Float32}:
 0.0
 0.0
```

Note that `init` creates the weight array, but not the bias vector.

Many of the initialisation functions accept keywords such as `gain`, 
and a random number generator. To make it easy to pass these to layers,
there are methods which return a function:

```jldoctest; setup = :(using Flux, Random)
julia> Dense(4 => 5, tanh; init=Flux.glorot_uniform(gain=2))
Dense(4 => 5, tanh)  # 25 parameters

julia> Dense(4 => 5, tanh; init=Flux.randn32(MersenneTwister(1)))
Dense(4 => 5, tanh)  # 25 parameters
```

## Initialisation functions

```@docs
Flux.glorot_uniform
Flux.glorot_normal
Flux.kaiming_uniform
Flux.kaiming_normal
Flux.truncated_normal
Flux.lecun_normal
Flux.orthogonal
Flux.sparse_init
Flux.identity_init
Flux.ones32
Flux.zeros32
Flux.rand32
Flux.randn32
Flux.create_bias
```

These functions call:

```@docs
Flux.rng_from_array
Flux.nfan
```

## Changing the type of all parameters

The default `eltype` for models is `Float32` since models are often trained/run on GPUs.
The `eltype` of model `m` can be changed to `Float64` by `f64(m)`:

```@docs
Flux.f64
Flux.f32
Flux.f16
```
