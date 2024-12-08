<p align="center">
    <img width="400px" src="https://raw.githubusercontent.com/FluxML/Flux.jl/master/docs/src/assets/logo.png#gh-light-mode-only"/>
    <img width="400px" src="https://raw.githubusercontent.com/FluxML/Flux.jl/master/docs/src/assets/logo-dark.png#gh-dark-mode-only"/>
</p>

<div align="center">

[![](https://img.shields.io/badge/Documentation-stable-blue.svg)](https://fluxml.github.io/Flux.jl/stable/) 
[![](https://img.shields.io/badge/Documentation-dev-blue.svg)](https://fluxml.github.io/Flux.jl/dev/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.00602/status.svg)](https://doi.org/10.21105/joss.00602) [![Flux Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FFlux&query=total_requests&suffix=%2Fmonth&label=Downloads)](http://juliapkgstats.com/pkg/Flux)
<br/>
[![][action-img]][action-url] [![][codecov-img]][codecov-url] [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

</div>

[action-img]: https://github.com/FluxML/Flux.jl/workflows/CI/badge.svg
[action-url]: https://github.com/FluxML/Flux.jl/actions
[codecov-img]: https://codecov.io/gh/FluxML/Flux.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/FluxML/Flux.jl

Flux is an elegant approach to machine learning. It's a 100% pure-Julia stack, and provides lightweight abstractions on top of Julia's native GPU and AD support. Flux makes the easy things easy while remaining fully hackable.

Works best with [Julia 1.10](https://julialang.org/downloads/) or later. Here's a very short example to try it out:
```julia
using Flux
data = [(x, 2x-x^3) for x in -2:0.1f0:2]

model = let
  w, b, v = (randn(Float32, 23) for _ in 1:3)  # parameters
  x -> sum(v .* tanh.(w*x .+ b))               # callable
end
# model = Chain(vcat, Dense(1 => 23, tanh), Dense(23 => 1, bias=false), only)

opt_state = Flux.setup(Adam(), model)
for epoch in 1:100
  Flux.train!((m,x,y) -> (m(x) - y)^2, model, data, opt_state)
end

using Plots
plot(x -> 2x-x^3, -2, 2, label="truth")
scatter!(model, -2:0.1f0:2, label="learned")
```
In Flux 0.15, almost any parameterised function in Julia is a valid Flux model -- such as this closure over `w, b, v`. The same function can also be implemented with built-in layers as shown.

The [quickstart page](https://fluxml.ai/Flux.jl/stable/guide/models/quickstart/) has a longer example. See the [documentation](https://fluxml.github.io/Flux.jl/) for details, or the [model zoo](https://github.com/FluxML/model-zoo/) for examples. Ask questions on the [Julia discourse](https://discourse.julialang.org/) or [slack](https://discourse.julialang.org/t/announcing-a-julia-slack/4866).

If you use Flux in your research, please [cite](CITATION.bib) our work.
