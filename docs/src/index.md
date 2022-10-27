# Flux: The Julia Machine Learning Library

Flux is a library for machine learning. It comes "batteries-included" with many useful tools built in, but also lets you use the full power of the Julia language where you need it. We follow a few key principles:

* **Doing the obvious thing**. Flux has relatively few explicit APIs for features like regularisation or embeddings. Instead, writing down the mathematical form will work – and be fast.
* **Extensible by default**. Flux is written to be highly extensible and flexible while being performant. Extending Flux is as simple as using your own code as part of the model you want - it is all [high-level Julia code](https://github.com/FluxML/Flux.jl/blob/ec16a2c77dbf6ab8b92b0eecd11661be7a62feef/src/layers/recurrent.jl#L131). When in doubt, it’s well worth looking at [the source](https://github.com/FluxML/Flux.jl/tree/master/src). If you need something different, you can easily roll your own.
* **Play nicely with others**. Flux works well with Julia libraries from [images](https://github.com/JuliaImages/Images.jl) to [differential equation solvers](https://github.com/SciML/DifferentialEquations.jl), so you can easily build complex data processing pipelines that integrate Flux models.

## Installation

Download [Julia 1.6](https://julialang.org/downloads/) or later, preferably the current stable release. You can add Flux using Julia's package manager, by typing `] add Flux` in the Julia prompt.

This will automatically install several other packages, including [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) which supports Nvidia GPUs. To directly access some of its functionality, you may want to add `] add CUDA` too. The page on [GPU support](gpu.md) has more details.

Other closely associated packages, also installed automatically, include [Zygote](https://github.com/FluxML/Zygote.jl), [Optimisers](https://github.com/FluxML/Optimisers.jl), [NNlib](https://github.com/FluxML/NNlib.jl), [Functors](https://github.com/FluxML/Functors.jl) and [MLUtils](https://github.com/JuliaML/MLUtils.jl).

## Learning Flux

The [quick start](@ref man-quickstart) page trains a simple neural network.

This rest of this documentation provides a from-scratch introduction to Flux's take on models and how they work, starting with [fitting a line](@ref man-overview). Once you understand these docs, congratulations, you also understand [Flux's source code](https://github.com/FluxML/Flux.jl), which is intended to be concise, legible and a good reference for more advanced concepts.

Sections with 📚 contain API listings. The same text is avalable at the Julia prompt, by typing for example `?gpu`.

If you just want to get started writing models, the [model zoo](https://github.com/FluxML/model-zoo/) gives good starting points for many common ones.

## Community

Everyone is welcome to join our community on the [Julia discourse forum](https://discourse.julialang.org/), or the [slack chat](https://discourse.julialang.org/t/announcing-a-julia-slack/4866) (channel #machine-learning). If you have questions or issues we'll try to help you out.

If you're interested in hacking on Flux, the [source code](https://github.com/FluxML/Flux.jl) is open and easy to understand -- it's all just the same Julia code you work with normally. You might be interested in our [intro issues](https://github.com/FluxML/Flux.jl/labels/good%20first%20issue) to get started, or our [contributing guide](https://github.com/FluxML/Flux.jl/blob/master/CONTRIBUTING.md).
