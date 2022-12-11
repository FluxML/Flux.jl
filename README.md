<p align="center">
    <img width="400px" src="https://raw.githubusercontent.com/FluxML/Flux.jl/master/docs/src/assets/logo.png#gh-light-mode-only"/>
    <img width="400px" src="https://raw.githubusercontent.com/FluxML/Flux.jl/master/docs/src/assets/logo-dark.png#gh-dark-mode-only"/>
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

Works best with [Julia 1.8](https://julialang.org/downloads/) or later. Here's a very short example to try it out:
```julia
using Flux, Plots
data = [([x], 2x-x^3) for x in -2:0.1f0:2]

model = Chain(Dense(1 => 23, tanh), Dense(23 => 1, bias=false), only)

optim = Flux.setup(Adam(), model)
for epoch in 1:1000
  Flux.train!((m,x,y) -> (m(x) - y)^2, model, data, optim)
end

plot(x -> 2x-x^3, -2, 2, legend=false)
scatter!(-2:0.1:2, [model([x]) for x in -2:0.1:2])
```

The [quickstart page](https://fluxml.ai/Flux.jl/stable/models/quickstart/) has a longer example. See the [documentation](https://fluxml.github.io/Flux.jl/) for details, or the [model zoo](https://github.com/FluxML/model-zoo/) for examples. Ask questions on the [Julia discourse](https://discourse.julialang.org/) or [slack](https://discourse.julialang.org/t/announcing-a-julia-slack/4866).

If you use Flux in your research, please [cite](CITATION.bib) our work.
