# Flux: The Julia Machine Learning Library

Flux is a library for machine learning. It comes "batteries-included" with many useful tools built in, but also lets you use the full power of the Julia language where you need it. We follow a few key principles:

* **Doing the obvious thing**. Flux has relatively few explicit APIs. Instead, writing down the mathematical form will work – and be fast.
* **Extensible by default**. Flux is written to be highly flexible while being performant. Extending Flux is as simple as using your own code as part of the model you want - it is all [high-level Julia code](https://github.com/FluxML/Flux.jl/tree/master/src).
* **Play nicely with others**. Flux works well with unrelated Julia libraries from [images](https://github.com/JuliaImages/Images.jl) to [differential equation solvers](https://github.com/SciML/DifferentialEquations.jl), rather than duplicating them.

### Installation

Download [Julia 1.6](https://julialang.org/downloads/) or later, preferably the current stable release. You can add Flux using Julia's package manager, by typing `] add Flux` in the Julia prompt. This will automatically install several other packages, including [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for Nvidia GPU support.

### Learning Flux

The **[quick start](@ref man-quickstart)** page trains a simple neural network.

This rest of the **guide** provides a from-scratch introduction to Flux's take on models and how they work, starting with [fitting a line](@ref man-overview). Once you understand these docs, congratulations, you also understand [Flux's source code](https://github.com/FluxML/Flux.jl), which is intended to be concise, legible and a good reference for more advanced concepts.

There are some **tutorials** about building particular models. The **[model zoo](https://github.com/FluxML/model-zoo/)** has starting points for many other common ones. And finally, the **[ecosystem page](ecosystem.md)** lists packages which define Flux models.

The **reference** section includes, beside Flux's own functions, those of some companion packages: [Zygote.jl](https://github.com/FluxML/Zygote.jl) (automatic differentiation), [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) (training) and others.

### Community

Everyone is welcome to join our community on the [Julia discourse forum](https://discourse.julialang.org/), or the [slack chat](https://discourse.julialang.org/t/announcing-a-julia-slack/4866) (channel #machine-learning). If you have questions or issues we'll try to help you out.

If you're interested in hacking on Flux, the [source code](https://github.com/FluxML/Flux.jl) is open and easy to understand -- it's all just the same Julia code you work with normally. You might be interested in our [intro issues](https://github.com/FluxML/Flux.jl/labels/good%20first%20issue) to get started, or our [contributing guide](https://github.com/FluxML/Flux.jl/blob/master/CONTRIBUTING.md).
