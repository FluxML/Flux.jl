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

Works best with [Julia 1.8](https://julialang.org/downloads/) or later. Here's a simple example to try it out:

```julia
using Flux

# We wish to learn this function:
f(x) = cos(x[1] * 5) - 0.2 * x[2]

# Generate dataset:
n = 10000
X = rand(2, n)  # In Julia, the batch axis is last!
Y = [f(X[:, i]) for i=1:n]
Y = reshape(Y, 1, n)

# Move to GPU (no need to delete; this is a no-op if you don't have one)
X = gpu(X)
Y = gpu(Y)

# Create dataloader
loader = Flux.DataLoader((X, Y), batchsize=64, shuffle=true)

# Create a simple fully-connected network (multi-layer perceptron):
n_in = 2
n_out = 1
model = Chain(
    Dense(n_in, 32), relu,
    Dense(32, 32), relu,
    Dense(32, 32), relu,
    Dense(32, n_out)
)
model = gpu(model)

# Create our optimizer:
optim = Adam(1e-3)
p = Flux.params(model)

# Let's train for 10 epochs:
for i in 1:10
    losses = []
    for (x, y) in loader
    
        # Compute gradient of the following code
        # with respect to parameters:
        loss, grad = Flux.withgradient(p) do
            # Forward pass:
            y_pred = model(x)
    
            # Square error loss
            sum((y_pred .- y) .^ 2)
        end
    
        # Step with this gradient:
        Flux.update!(optim, p, grad)

        # Logging:
        push!(losses, loss)
    end
    println(sum(losses)/length(losses))
end
```

We can visualize our predictions with `Plots.jl`:

```julia
using Plots

# Generate test dataset:
Xtest = rand(2, 100)
Ytest = mapslices(f, Xtest; dims=1)  # Alternative syntax to apply the function `f`

# View the predictions:
Ypredicted = model(Xtest)
scatter(Ytest[1, :], Ypredicted[1, :], xlabel="true", ylabel="predicted")
```

See the [documentation](https://fluxml.github.io/Flux.jl/) for details, or the [model zoo](https://github.com/FluxML/model-zoo/) for examples. Ask questions on the [Julia discourse](https://discourse.julialang.org/) or [slack](https://discourse.julialang.org/t/announcing-a-julia-slack/4866).

If you use Flux in your research, please [cite](CITATION.bib) our work.
