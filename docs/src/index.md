# Flux

Flux is a high-level interface for machine learning, implemented in Julia.

Flux aims to be an intuitive and powerful notation, close to the mathematics, that provides advanced features like auto-unrolling and closures. Simple models are trivial, while the most complex architectures are tractable, taking orders of magnitude less code than in other frameworks. Meanwhile, the Flux compiler provides excellent error messages and tools for debugging when things go wrong.

So what's the catch? Flux is at an early "working prototype" stage; many things work but the API is still in a state of... well, it might change. Also, this documentation is pretty incomplete.

If you're interested to find out what *does* work, read on!

## Installation

*... Charging Ion Capacitors ...*

```julia
Pkg.clone("https://github.com/MikeInnes/DataFlow.jl")
Pkg.clone("https://github.com/MikeInnes/Flux.jl")
using Flux
```

You'll also need a backend to run real training, if you don't have one already. Choose from [MXNet](https://github.com/dmlc/MXNet.jl) or [TensorFlow](https://github.com/malmaud/TensorFlow.jl) (MXNet is the recommended option if you're not sure):

```julia
Pkg.add("MXNet") # or "TensorFlow"
```
