<p align="center">
<img width="400px" src="https://raw.githubusercontent.com/FluxML/fluxml.github.io/master/logo.png"/>
</p>

<div align="center">

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://fluxml.github.io/Flux.jl/stable/) [![](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://julialang.org/slack/) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) [![DOI](https://joss.theoj.org/papers/10.21105/joss.00602/status.svg)](https://doi.org/10.21105/joss.00602)
<br/>
[![][action-img]][action-url] [![][codecov-img]][codecov-url]

</div>

[action-img]: https://github.com/FluxML/Flux.jl/workflows/CI/badge.svg
[action-url]: https://github.com/FluxML/Flux.jl/actions
[codecov-img]: https://codecov.io/gh/FluxML/Flux.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/FluxML/Flux.jl

Flux is an elegant approach to machine learning. It's a 100% pure-Julia stack, and provides lightweight abstractions on top of Julia's native GPU and AD support. Flux makes the easy things easy while remaining fully hackable.

Works best with [Julia 1.8](https://julialang.org/downloads/) or later. This will install everything (including CUDA) and solve the XOR problem:
```julia
using Flux

x = hcat(digits.(0:3, base=2, pad=2)...) |> gpu
y = Flux.onehotbatch(xor.(eachrow(x)...), 0:1) |> gpu
data = ((Float32.(x), y) for _ in 1:100)

model = Chain(Dense(2 => 3, sigmoid), BatchNorm(3), Dense(3 => 2), softmax) |> gpu
optim = Adam(0.1, (0.7, 0.95))
loss(x, y) = Flux.crossentropy(model(x), y)

Flux.train!(loss, Flux.params(model), data, optim)

all((model(x) .> 0.5) .== y)
```

See the [documentation](https://fluxml.github.io/Flux.jl/) for details, the [website](https://fluxml.ai/tutorials.html) for tutorials, or the [model zoo](https://github.com/FluxML/model-zoo/) for examples.

If you use Flux in your research, please [cite](CITATION.bib) our work.
