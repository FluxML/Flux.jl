# GPU Support

Support for array operations on other hardware backends, like GPUs, is provided by external packages like [CuArrays](https://github.com/JuliaGPU/CuArrays.jl) and [CLArrays](https://github.com/JuliaGPU/CLArrays.jl). Flux doesn't care what array type you use, so we can just plug these in without any other changes.

For example, we can use `CuArrays` (with the `cu` converter) to run our [basic example](models/basics.md) on an NVIDIA GPU.

```julia
using CuArrays

W = cu(rand(2, 5)) # a 2×5 CuArray
b = cu(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = cu(rand(5)), cu(rand(2)) # Dummy data
loss(x, y) # ~ 3
```

Note that we convert both the parameters (`W`, `b`) and the data set (`x`, `y`) to cuda arrays. Taking derivatives and training works exactly as before.

If you define a structured model, like a `Dense` layer or `Chain`, you just need to convert the internal parameters. Flux provides `mapleaves`, which allows you to alter all parameters of a model at once.

```julia
d = Dense(10, 5, σ)
d = mapleaves(cu, d)
d.W # Tracked CuArray
d(cu(rand(10))) # CuArray output

m = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
m = mapleaves(cu, m)
d(cu(rand(10)))
```

The [mnist example](https://github.com/FluxML/model-zoo/blob/master/mnist/mlp.jl) contains the code needed to run the model on the GPU; just uncomment the lines after `using CuArrays`.
