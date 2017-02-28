# Loading and Save Models

```julia
model = Chain(Affine(10, 20), σ, Affine(20, 15), softmax)
```

Since models are just simple Julia data structures, it's very easy to save and load them using any of Julia's existing serialisation formats. For example, using Julia's built-in `serialize`:

```julia
open(io -> serialize(io, model), "model.jls", "w")
open(io -> deserialize(io), "model.jls")
```

One issue with `serialize` is that it doesn't promise compatibility between major Julia versions. For longer-term storage it's good to use a package like [JLD](https://github.com/JuliaIO/JLD.jl).

```julia
using JLD
@save "model.jld" model
@load "model.jld"
```

However, JLD will break for some models as functions are not supported on 0.5+. You can resolve that by checking out [this branch](https://github.com/JuliaIO/JLD.jl/pull/137).

Right now this is the only storage format Flux supports. In future Flux will support loading and saving other model formats (on an as-needed basis).
