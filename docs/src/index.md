# Flux: The Julia Machine Learning Library

Flux is a machine learning library for Julia. It gives you layers, losses and training tools that you can compose with your own Julia code.

- [Train your first neural network](@ref man-quickstart) with the Flux quick start.
- [Compile training and inference with Reactant](guide/reactant.md) on CPU and supported accelerators. For training, Flux compiles the forward pass, gradient calculation and optimiser update together.
- [Load pretrained models with Metalhead.jl](@ref man-pretrained-models) for vision, language and image generation.

### Installation

Download [Julia 1.10](https://julialang.org/downloads/) or later, preferably the current stable release. You can add Flux using Julia's package manager by typing `] add Flux` in the Julia prompt.
For hardware-specific setup, see the [GPU guide](guide/gpu.md).

### Learning Flux

The **guide** introduces Flux from scratch, starting with [fitting a line](@ref man-overview). It covers how to define models, compute gradients and write training loops.

The **tutorials** walk through specific tasks. For complete training scripts, see [Models and Examples](tutorials/model_zoo.md). For libraries that work with Flux, see the [ecosystem page](ecosystem.md).

The **reference** section includes, beside Flux's own functions, those of some companion packages: [Zygote.jl](https://github.com/FluxML/Zygote.jl) (automatic differentiation), [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) (training) and others.

### Community

Everyone is welcome to join our community on the [Julia discourse forum](https://discourse.julialang.org/), or the [slack chat](https://discourse.julialang.org/t/announcing-a-julia-slack/4866) (channel #machine-learning). If you have questions or issues we'll try to help you out.

If you're interested in hacking on Flux, the [source code](https://github.com/FluxML/Flux.jl) is open and easy to understand -- it's all just the same Julia code you work with normally. You might be interested in our [intro issues](https://github.com/FluxML/Flux.jl/labels/good%20first%20issue) to get started, or our [contributing guide](https://github.com/FluxML/Flux.jl/blob/master/CONTRIBUTING.md).
