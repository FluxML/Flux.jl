# Flux: The Julia Machine Learning Library

Flux is a library for machine learning. It comes "batteries-included" with many useful tools built in, but also lets you use the full power of the Julia language where you need it. The whole stack is implemented in clean Julia code (right down to the [GPU kernels](https://github.com/FluxML/CuArrays.jl)) and any part can be tweaked to your liking.

# Installation

Install [Julia 0.6.0 or later](https://julialang.org/downloads/), if you haven't already.

```julia
Pkg.add("Flux")
# Optional but recommended
Pkg.update() # Keep your packages up to date
Pkg.test("Flux") # Check things installed correctly
```

Start with the [basics](models/basics.md). The [model zoo](https://github.com/FluxML/model-zoo/) is also a good starting point for many common kinds of models.
