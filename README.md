<p align="center">
    <img width="400px" src="https://raw.githubusercontent.com/FluxML/Flux.jl/master/docs/src/assets/logo.png#gh-light-mode-only"/>
    <img width="400px" src="https://raw.githubusercontent.com/FluxML/Flux.jl/master/docs/src/assets/logo-dark.png#gh-dark-mode-only"/>
</p>

<div align="center">

[![](https://img.shields.io/badge/Documentation-stable-blue.svg)](https://fluxml.ai/Flux.jl/stable/) 
[![](https://img.shields.io/badge/Documentation-dev-blue.svg)](https://fluxml.ai/Flux.jl/dev/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.00602/status.svg)](https://doi.org/10.21105/joss.00602) [![Flux Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FFlux&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/Flux)
<br/>
[![][action-img]][action-url] [![][codecov-img]][codecov-url] [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

</div>

[action-img]: https://github.com/FluxML/Flux.jl/workflows/CI/badge.svg
[action-url]: https://github.com/FluxML/Flux.jl/actions
[codecov-img]: https://codecov.io/gh/FluxML/Flux.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/FluxML/Flux.jl

Flux is an elegant approach to machine learning. It's a 100% pure-Julia stack, and provides lightweight abstractions on top of Julia's native GPU and AD support. Flux makes the easy things easy while remaining fully hackable.

- **Layers and losses** for MLPs, convolutional, recurrent and attention models, with normalisation, pooling, embeddings and more.
- **Your choice of AD engine**: [Zygote](https://github.com/FluxML/Zygote.jl) by default, or [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) and [Mooncake](https://github.com/chalk-lab/Mooncake.jl), see [gradients](https://fluxml.ai/Flux.jl/stable/reference/training/gradients/).
- **GPU support** for NVIDIA, AMD and Apple hardware, see the [GPU guide](https://fluxml.ai/Flux.jl/stable/guide/gpu/).
- **Compilation with [Reactant](https://fluxml.ai/Flux.jl/stable/guide/reactant/)**: the whole training step fused into one XLA program, on CPU, GPU or TPU.
- **Distributed training** with MPI and NCCL, see the [GPU guide](https://fluxml.ai/Flux.jl/stable/guide/gpu/#Distributed-data-parallel-training).

Works best with [Julia 1.10](https://julialang.org/downloads/) or later. Install Flux from the Julia REPL with `using Pkg; Pkg.add("Flux")`, then try this very short example:
```julia
using Flux

# XOR problem: classify the points of the unit square by quadrant
noisy = rand(Float32, 2, 1000)  # one point per column
truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]
target = Flux.onehotbatch(truth, [true, false])
loader = Flux.DataLoader((noisy, target), batchsize=64, shuffle=true)

model = Chain(Dense(2 => 3, tanh), BatchNorm(3), Dense(3 => 2))
opt_state = Flux.setup(Adam(0.01), model)

for epoch in 1:1_000
  Flux.train!(model, loader, opt_state) do m, x, y
    Flux.logitcrossentropy(m(x), y)
  end
end

accuracy = sum(Flux.onecold(model(noisy), [true, false]) .== truth) / length(truth)
```

To train on a GPU, load the package for your hardware and move the model and the data loader to the device:
```julia
using CUDA                  # or AMDGPU, Metal
device = gpu_device()       # the CPU, if no GPU is available
model = model |> device     # before calling Flux.setup
loader = loader |> device   # moves one batch at a time
```

Flux models are plain Julia. In fact, almost any parameterised function is a valid model, such as the closure over `w, b, v` below. The same function can also be implemented with built-in layers as shown.
```julia
using Flux
data = [(x, 2x-x^3) for x in -2:0.1f0:2]

model = let
  w, b, v = (randn(Float32, 23) for _ in 1:3)  # parameters
  x -> sum(v .* tanh.(w*x .+ b))               # callable
end
# model = Chain(vcat, Dense(1 => 23, tanh), Dense(23 => 1, bias=false), only)

opt_state = Flux.setup(Adam(), model)
for epoch in 1:300
  Flux.train!((m,x,y) -> (m(x) - y)^2, model, data, opt_state)
end

using Plots
plot(x -> 2x-x^3, -2, 2, label="truth")
scatter!(model, -2:0.1f0:2, label="learned")
```

The [quickstart page](https://fluxml.ai/Flux.jl/stable/guide/models/quickstart/) has a longer version of the first example, with plots. See the [documentation](https://fluxml.ai/Flux.jl/) for details, the [model zoo](https://github.com/FluxML/model-zoo/) for examples, or the [examples](examples/) folder for larger end-to-end scripts: [GPT-2 pretraining](examples/gpt2_fineweb) and [ResNet on TinyImageNet](examples/resnet_tinyimagenet). Ask questions on the [Julia discourse](https://discourse.julialang.org/) or [slack](https://discourse.julialang.org/t/announcing-a-julia-slack/4866).

If you use Flux in your research, please [cite](CITATION.bib) our work.
