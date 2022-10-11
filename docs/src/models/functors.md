# Recursive transformations from Functors.jl

Flux models are deeply nested structures, and [Functors.jl](https://github.com/FluxML/Functors.jl) provides tools needed to explore such objects, apply functions to the parameters they contain, and re-build them.

New layers should be annotated using the `Functors.@functor` macro. This will enable [`params`](@ref Flux.params) to see the parameters inside, and [`gpu`](@ref) to move them to the GPU.

`Functors.jl` has its own [notes on basic usage](https://fluxml.ai/Functors.jl/stable/#Basic-Usage-and-Implementation) for more details. Additionally, the [Advanced Model Building and Customisation](../models/advanced.md) page covers the use cases of `Functors` in greater details.

```@docs
Functors.@functor
Functors.fmap
Functors.isleaf
Functors.children
Functors.fcollect
Functors.functor
Functors.fmapstructure
```
