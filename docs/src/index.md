# Flux: The Julia Machine Learning Library

Flux is a library for machine learning geared towards high-performance production pipelines. It comes "batteries-included" with many useful tools built in, but also lets you use the full power of the Julia language where you need it. We follow a few key principles:

* **Doing the obvious thing**. Flux has relatively few explicit APIs for features like regularisation or embeddings. Instead, writing down the mathematical form will work – and be fast.
* **Extensible by default**. Flux is written to be highly extensible and flexible while being performant. Extending Flux is as simple as using your own code as part of the model you want - it is all [high-level Julia code](https://github.com/FluxML/Flux.jl/blob/ec16a2c77dbf6ab8b92b0eecd11661be7a62feef/src/layers/recurrent.jl#L131). When in doubt, it’s well worth looking at [the source](https://github.com/FluxML/Flux.jl/). If you need something different, you can easily roll your own.
* **Performance is key**. Flux integrates with high-performance AD tools such as [Zygote.jl](https://github.com/FluxML/Zygote.jl) for generating fast code. Flux optimizes both CPU and GPU performance. Scaling workloads easily to multiple GPUs can be done with the help of Julia's [GPU tooling](https://github.com/JuliaGPU/CUDA.jl) and projects like [DaggerFlux.jl](https://github.com/DhairyaLGandhi/DaggerFlux.jl).
* **Play nicely with others**. Flux works well with Julia libraries from [data frames](https://github.com/JuliaComputing/JuliaDB.jl) and [images](https://github.com/JuliaImages/Images.jl) to [differential equation solvers](https://github.com/JuliaDiffEq/DifferentialEquations.jl), so you can easily build complex data processing pipelines that integrate Flux models.

## Installation

Download [Julia 1.6](https://julialang.org/) or later, if you haven't already. You can add Flux using Julia's package manager, by typing `] add Flux` in the Julia prompt.

If you have CUDA you can also run `] add CUDA` to get GPU support; see [here](gpu.md) for more details.

NOTE: Flux used to have a CuArrays.jl dependency until v0.10.4, replaced by CUDA.jl in v0.11.0. If you're upgrading Flux from v0.10.4 or a lower version, you may need to remove CuArrays (run `] rm CuArrays`) before you can upgrade.

## Learning Flux

There are several different ways to learn Flux. If you just want to get started writing models, the [model zoo](https://github.com/FluxML/model-zoo/) gives good starting points for many common ones. This documentation provides a reference to all of Flux's APIs, as well as a from-scratch introduction to Flux's take on models and how they work. Once you understand these docs, congratulations, you also understand [Flux's source code](https://github.com/FluxML/Flux.jl), which is intended to be concise, legible and a good reference for more advanced concepts.
