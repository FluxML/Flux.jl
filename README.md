<p align="center">
<img width="400px" src="https://raw.githubusercontent.com/FluxML/fluxml.github.io/master/logo.png"/>
</p>

<div align="center">

[![](https://img.shields.io/badge/Documentation-stable-blue.svg)](https://fluxml.github.io/Flux.jl/stable/) [![DOI](https://joss.theoj.org/papers/10.21105/joss.00602/status.svg)](https://doi.org/10.21105/joss.00602) [![Flux Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/Flux)](https://pkgs.genieframework.com?packages=Flux)
<br/>
[![][action-img]][action-url] [![][codecov-img]][codecov-url] [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

</div>

[action-img]: https://github.com/FluxML/Flux.jl/workflows/CI/badge.svg
[action-url]: https://github.com/FluxML/Flux.jl/actions
[codecov-img]: https://codecov.io/gh/FluxML/Flux.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/FluxML/Flux.jl

Flux is an elegant approach to machine learning. It's a 100% pure-Julia stack, and provides lightweight abstractions on top of Julia's native GPU and AD support. Flux makes the easy things easy while remaining fully hackable.

Works best with [Julia 1.8](https://julialang.org/downloads/) or later. Here's a simple example to try it out:
```julia
using Flux  # should install everything for you, including CUDA

x = hcat(digits.(0:3, base=2, pad=2)...) |> gpu  # let's solve the XOR problem!
y = Flux.onehotbatch(xor.(eachrow(x)...), 0:1) |> gpu
data = ((Float32.(x), y) for _ in 1:100)  # an iterator making Tuples

model = Chain(Dense(2 => 3, sigmoid), BatchNorm(3), Dense(3 => 2), softmax) |> gpu
optim = Adam(0.1, (0.7, 0.95))
loss(x, y) = Flux.crossentropy(model(x), y)

Flux.train!(loss, Flux.params(model), data, optim)  # updates model & optim

all((model(x) .> 0.5) .== y)  # usually 100% accuracy.
```

See the [documentation](https://fluxml.github.io/Flux.jl/) for details, or the [model zoo](https://github.com/FluxML/model-zoo/) for examples. Ask questions on the [Julia discourse](https://discourse.julialang.org/) or [slack](https://discourse.julialang.org/t/announcing-a-julia-slack/4866).

If you use Flux in your research, please [cite](CITATION.bib) our work.
