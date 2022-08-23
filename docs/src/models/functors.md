# Recursive transformations from Functors.jl

Flux models are deeply nested structures, and [Functors.jl](https://github.com/FluxML/Functors.jl) provides tools needed to explore such objects, apply functions to the parameters they contain, and re-build them.

Functors.jl is a collection of tools designed to represent a [functor](https://en.wikipedia.org/wiki/Functor_(functional_programming)). Flux makes use of it to treat certain structs as functors. Notable examples include the layers that Flux defines.

```@docs
Functors.isleaf
Functors.children
Functors.fcollect
Functors.functor
Functors.@functor
Functors.fmap
Functors.fmapstructure
```
