# Flux: The Julia Machine Learning Library

Flux is a library for machine learning. It comes "batteries-included" with many useful tools built in, but also lets you use the full power of the Julia language where you need it. We follow a few key principles:

* **Doing the obvious thing**. Flux has relatively few explicit APIs for features like regularisation or embeddings. Instead, writing down the mathematical form will work – and be fast.
* **You could have written Flux**. All of it, from [LSTMs](https://github.com/FluxML/Flux.jl/blob/ec16a2c77dbf6ab8b92b0eecd11661be7a62feef/src/layers/recurrent.jl#L131) to [GPU kernels](https://github.com/JuliaGPU/CuArrays.jl), is straightforward Julia code. When in doubt, it’s well worth looking at [the source](https://github.com/FluxML/Flux.jl/). If you need something different, you can easily roll your own.
* **Play nicely with others**. Flux works well with Julia libraries from [data frames](https://github.com/JuliaComputing/JuliaDB.jl) and [images](https://github.com/JuliaImages/Images.jl) to [differential equation solvers](https://github.com/JuliaDiffEq/DifferentialEquations.jl), so you can easily build complex data processing pipelines that integrate Flux models.

## Installation

Download [Julia 1.0](https://julialang.org/) or later, if you haven't already. You can add Flux from using Julia's package manager, by typing `] add Flux` in the Julia prompt.

If you have CUDA you can also run `] add CuArrays` to get GPU support; see [here](gpu.md) for more details.

## Learning Flux

There are several different ways to learn Flux. If you just want to get started writing models, the [model zoo](https://github.com/FluxML/model-zoo/) gives good starting points for many common ones. This documentation provides a reference to all of Flux's APIs, as well as a from-scratch introduction to Flux's take on models and how they work. Once you understand these docs, congratulations, you also understand [Flux's source code](https://github.com/FluxML/Flux.jl), which is intended to be concise, legible and a good reference for more advanced concepts.
