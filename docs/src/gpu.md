# GPU Support

NVIDIA GPU support should work out of the box on systems with CUDA and CUDNN installed. For more details see the [CUDA](https://github.com/JuliaGPU/CUDA.jl) readme.

## Checking GPU Availability

By default, Flux will run the checks on your system to see if it can support GPU functionality. You can check if Flux identified a valid GPU setup by typing the following:

```julia
julia> using CUDA

julia> CUDA.functional()
true
```

## GPU Usage

Support for array operations on other hardware backends, like GPUs, is provided by external packages like [CUDA](https://github.com/JuliaGPU/CUDA.jl). Flux is agnostic to array types, so we simply need to move model weights and data to the GPU and Flux will handle it.

For example, we can use `CUDA.CuArray` (with the `cu` converter) to run our [basic example](models/basics.md) on an NVIDIA GPU.

(Note that you need to have CUDA available to use CUDA.CuArray – please see the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) instructions for more details.)

```julia
using CUDA

W = cu(rand(2, 5)) # a 2×5 CuArray
b = cu(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = cu(rand(5)), cu(rand(2)) # Dummy data
loss(x, y) # ~ 3
```

Note that we convert both the parameters (`W`, `b`) and the data set (`x`, `y`) to cuda arrays. Taking derivatives and training works exactly as before.

If you define a structured model, like a `Dense` layer or `Chain`, you just need to convert the internal parameters. Flux provides `fmap`, which allows you to alter all parameters of a model at once.

```julia
d = Dense(10, 5, σ)
d = fmap(cu, d)
d.weight # CuArray
d(cu(rand(10))) # CuArray output

m = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
m = fmap(cu, m)
d(cu(rand(10)))
```

As a convenience, Flux provides the `gpu` function to convert models and data to the GPU if one is available. By default, it'll do nothing. So, you can safely call `gpu` on some data or model (as shown below), and the code will not error, regardless of whether the GPU is available or not. If the GPU library (CUDA.jl) loads successfully, `gpu` will move data from the CPU to the GPU. As is shown below, this will change the type of something like a regular array to a `CuArray`.

```julia
julia> using Flux, CUDA

julia> m = Dense(10,5) |> gpu
Dense(10, 5)

julia> x = rand(10) |> gpu
10-element CuArray{Float32,1}:
 0.800225
 ⋮
 0.511655

julia> m(x)
5-element CuArray{Float32,1}:
 -0.30535
 ⋮
 -0.618002
```

The analogue `cpu` is also available for moving models and data back off of the GPU.

```julia
julia> x = rand(10) |> gpu
10-element CuArray{Float32,1}:
 0.235164
 ⋮
 0.192538

julia> x |> cpu
10-element Array{Float32,1}:
 0.235164
 ⋮
 0.192538
```

## Disabling CUDA or choosing which GPUs are visible to Flux

Sometimes it is required to control which GPUs are visible to `julia` on a system with multiple GPUs or disable GPUs entirely. This can be achieved with an environment variable `CUDA_VISIBLE_DEVICES`.

To disable all devices:
```
$ export CUDA_VISIBLE_DEVICES='-1'
```
To select specific devices by device id:
```
$ export CUDA_VISIBLE_DEVICES='0,1'
```


More information for conditional use of GPUs in CUDA.jl can be found in its [documentation](https://cuda.juliagpu.org/stable/installation/conditional/#Conditional-use), and information about the specific use of the variable is described in the [Nvidia CUDA blogpost](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/).
