# Basic Usage

## Installation

```julia
Pkg.clone("https://github.com/MikeInnes/DataFlow.jl")
Pkg.clone("https://github.com/MikeInnes/Flux.jl")
```

## The Model

*Charging Ion Capacitors...*

The core concept in Flux is that of the *model*. A model is simply a function with parameters. In Julia, we might define the following function:

```julia
W = randn(3,5)
b = randn(3)
affine(x) = W*x + b

x1 = randn(5)
affine(x1)
> 3-element Array{Float64,1}:
   -0.0215644
   -4.07343  
    0.312591
```

## An MNIST Example
