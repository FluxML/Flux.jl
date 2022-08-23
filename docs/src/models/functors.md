# Recursive transformations from Functors.jl

Flux models are deeply nested structures, and [Functors.jl](https://github.com/FluxML/Functors.jl) provides tools needed to explore such objects, apply functions to the parameters they contain, and re-build them.

More precisely, using the `Functors.@functor` macro allows `Flux` layers to access additional functionalities, like collecting parameters or scaling them to the GPU.

Functors.jl is a collection of tools designed to represent a [functor](https://en.wikipedia.org/wiki/Functor_(functional_programming)). Flux makes use of it to treat certain structs as functors. Notable examples include the layers that Flux defines. The basic usage of `Functors.jl` can be found [here](https://fluxml.ai/Functors.jl/stable/#Basic-Usage-and-Implementation).

```@docs
Functors.@functor
Functors.fmap
Functors.isleaf
Functors.children
Functors.fcollect
Functors.functor
Functors.fmapstructure
```
