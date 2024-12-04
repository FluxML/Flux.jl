```@meta
CollapsedDocStrings = true
```

# Recursive transformations from Functors.jl

Flux models are deeply nested structures, and [Functors.jl](https://github.com/FluxML/Functors.jl) provides tools needed to explore such objects, apply functions to the parameters they contain (e.g. for moving them to gpu), and re-build them.

!!! compat "Flux ≤ v0.14"
    All layers were previously defined with the `Functors.@functor` macro.
    This still works, but it is recommended that you use the new [`Flux.@layer`](@ref Flux.@layer) macro instead.
    Both allow [`Flux.setup`](@ref Flux.setup) to see the parameters inside, and [`gpu`](@ref) to move them to the GPU, but [`Flux.@layer`](@ref Flux.@layer) also overloads printing,
    and offers a way to define `trainable` at the same time.

!!! compat "Functors v0.5"
    With Functors.jl v0.5, which is required by Flux v0.15 and later, every custom type is a functor by default. This means that applying `Flux.@layer` to a type is no longer strictly necessary, but it is still recommended for addictional features like pretty-printing.

`Functors.jl` has its own [notes on basic usage](https://fluxml.ai/Functors.jl/stable/#Basic-Usage-and-Implementation) for more details. Additionally, the [Advanced Model Building and Customisation](@ref man-advanced) page covers the use cases of `Functors` in greater details.

```@docs
Flux.@layer
Functors.@leaf
Functors.@functor
Functors.fmap
Functors.fmap_with_path
Functors.isleaf
Functors.children
Functors.fcollect
Functors.functor
Functors.fmapstructure
Functors.fmapstructure_with_path
Functors.execute
Functors.AbstractWalk
Functors.ExcludeWalk
Functors.CachedWalk
```

## Moving models, or data, to the GPU

Flux provides some convenience functions based on `fmap`. Some ([`f16`](@ref Flux.f16), [`f32`](@ref Flux.f32), [`f64`](@ref Flux.f64)) change the precision of all arrays in a model. Others are used for moving a model to of from GPU memory:

```@docs
cpu
gpu(::Any)
gpu(::Flux.DataLoader)
```
