<p align="center">
<img width="200px" src="https://raw.githubusercontent.com/FluxML/fluxml.github.io/master/flux.png"/>
</p>

[![Build Status](https://travis-ci.org/FluxML/Flux.jl.svg?branch=master)](https://travis-ci.org/FluxML/Flux.jl) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://fluxml.github.io/Flux.jl/stable/) [![](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://slackinvite.julialang.org/)

Flux is an elegant approach to machine learning. It's a 100% pure-Julia stack, and provides lightweight abstractions on top of Julia's native GPU and AD support. Flux makes the easy things easy while remaining fully hackable.

You need to build Julia 0.6 from source and have CUDA available to use Flux with GPUs – please see the [CUDAnative.jl](https://github.com/JuliaGPU/CUDAnative.jl) instructions for more details.

```julia
julia> Pkg.add("Flux")
```

See the [documentation](http://fluxml.github.io/Flux.jl/) or the [model zoo](https://github.com/FluxML/model-zoo/) for examples.

## Features

Flux has powerful high-level features, and common architectures can be defined in a few lines.

```julia
model = Chain(
  Dense(768, 128),
  LSTM(128, 256)
  LSTM(256, 128)
  Dense(128, 10),
  softmax)

loss(x, y) = crossentropy(model(x), y)

Flux.train!(loss, data, ADAM(...))
```

Yet you can easily strip away the layers, and directly write the mathematics for your problem. Flux will seamlessly take gradients of any Julia code, so your model looks just like the paper.

```julia
W = param(randn(2, 10))
b = param(randn(2))

y(x) = σ.(W * x .+ b)
```

If that's *still* not enough, you can go as deep as you want, even writing your own CUDA kernels with [CUDAnative](https://github.com/JuliaGPU/CUDAnative.jl)! All this can be freely mixed-and-matched in a single model or script, and it all runs interactively via Jupyter or Juno.

```julia
function gpu_add(a, b, c)
  i = (blockIdx().x-1) * blockDim().x + threadIdx().x
  c[i] = a[i] + b[i]
  return nothing
end
```

Unusual architectures are no problem in Flux, as you can use all the loops, control flow and even macros that you're used to. Here's a Tree RNN in 4 lines.

```julia
tree() = rand() < 0.5 ? rand(10) : (tree(), tree()) # dummy data

shrink = Dense(20, 10)
combine(a, b) = shrink([a; b])

model(x) = x
model(x::Tuple) = combine(model(x[1]), model(x[2]))

model(tree()) # Sample output
```

Despite this flexibility, Julia's advanced compiler lets us do some powerful optimisations. For example, this definition of `sigmoid` automatically gets fused into a *single* GPU kernel – so it's really fast.

```julia
sigmoid(xs) = 1 ./ (1 .+ exp.(.-xs))
```

Similarly, Flux is the first dynamic framework to support [compiling to the browser](https://fluxml.github.io/experiments/) and model import via [formats like ONNX](https://github.com/FluxML/ONNX.jl/), both of which are thinly-veiled compiler problems.

For more on our philosophy on machine learning, check out our article [On Machine Learning & Programming Languages](https://julialang.org/blog/2017/12/ml&pl).

## Contributing & Help

For general questions and help, check out Julia's [community forum](https://discourse.julialang.org/c/domain/ML).

Flux development is carried out via our [GitHub issues](https://github.com/FluxML/Flux.jl/issues), so feel free to open feature requests or PRs here.

For more informal discussions we'd love to have you on the [Julia slack](https://slackinvite.julialang.org/), where we hang out on the #machine-learning channel.
