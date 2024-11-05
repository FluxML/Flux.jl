
# [Automatic Differentiation using Enzyme.jl](@id autodiff-enzyme)

Flux now builds in support for Enzyme.jl

```@docs
gradient(f, args::Union{EnzymeCore.Const, EnzymeCore.Duplicated}...)
Flux.withgradient(f, args::Union{EnzymeCore.Const, EnzymeCore.Duplicated}...)
```
