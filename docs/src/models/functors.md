## Functors.jl

Flux makes use of the [Functors.jl](https://github.com/FluxML/Functors.jl) to represent many of the core functionalities it provides.

Functors.jl is a collection of tools designed to represent a [functor](https://en.wikipedia.org/wiki/Functor_(functional_programming)). Flux makes use of it to treat certain structs as functors. Notable examples include the layers that Flux defines.

```@docs
Functors.isleaf
Functors.children
Functors.fcollect
Functors.functor
Functors.fmap
```
