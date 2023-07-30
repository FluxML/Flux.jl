# GPU Support

Starting with v0.14, Flux doesn't force a specific GPU backend and the corresponding package dependencies on the users. 
Thanks to the [package extension mechanism](
https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)) introduced in julia v1.9, Flux conditionally loads GPU specific code once a GPU package is made available (e.g. through `using CUDA`).

NVIDIA GPU support requires the packages `CUDA.jl` and `cuDNN.jl` to be installed in the environment. In the julia REPL, type `] add CUDA, cuDNN` to install them. For more details see the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) readme.

AMD GPU support is available since Julia 1.9 on systems with ROCm and MIOpen installed. For more details refer to the [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) repository.

Metal GPU acceleration is available on Apple Silicon hardware. For more details refer to the [Metal.jl](https://github.com/JuliaGPU/Metal.jl) repository. Metal support in Flux is experimental and many features are not yet available.

In order to trigger GPU support in Flux, you need to call `using CUDA`, `using AMDGPU` or `using Metal`
in your code. Notice that for CUDA, explicitly loading also `cuDNN` is not required, but the package has to be installed in the environment. 


!!! compat "Flux ≤ 0.13"
    Old versions of Flux automatically installed CUDA.jl to provide GPU support. Starting from Flux v0.14, CUDA.jl is not a dependency anymore and has to be installed manually.

## Checking GPU Availability

By default, Flux will run the checks on your system to see if it can support GPU functionality. You can check if Flux identified a valid GPU setup by typing the following:

```julia
julia> using CUDA

julia> CUDA.functional()
true
```

For AMD GPU:

```julia
julia> using AMDGPU

julia> AMDGPU.functional()
true

julia> AMDGPU.functional(:MIOpen)
true
```

For Metal GPU:

```julia
julia> using Metal

julia> Metal.functional()
true
```

## Selecting GPU backend

Available GPU backends are: `CUDA`, `AMD` and `Metal`.

Flux relies on [Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl) for selecting default GPU backend to use.

There are two ways you can specify it:

- From the REPL/code in your project, call `Flux.gpu_backend!("AMD")` and restart (if needed) Julia session for the changes to take effect.
- In `LocalPreferences.toml` file in you project directory specify:
```toml
[Flux]
gpu_backend = "AMD"
```

Current GPU backend can be fetched from `Flux.GPU_BACKEND` variable:

```julia
julia> Flux.GPU_BACKEND
"CUDA"
```

The current backend will affect the behaviour of methods like the method `gpu` described below.

## Basic GPU Usage

Support for array operations on other hardware backends, like GPUs, is provided by external packages like [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), and [Metal.jl](https://github.com/JuliaGPU/Metal.jl).
Flux is agnostic to array types, so we simply need to move model weights and data to the GPU and Flux will handle it.

For example, we can use `CUDA.CuArray` (with the `cu` converter) to run our [basic example](@ref man-basics) on an NVIDIA GPU.

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
d = Dense(10 => 5, σ)
d = fmap(cu, d)
d.weight # CuArray
d(cu(rand(10))) # CuArray output

m = Chain(Dense(10 => 5, σ), Dense(5 => 2), softmax)
m = fmap(cu, m)
m(cu(rand(10)))
```

As a convenience, Flux provides the `gpu` function to convert models and data to the GPU if one is available. By default, it'll do nothing. So, you can safely call `gpu` on some data or model (as shown below), and the code will not error, regardless of whether the GPU is available or not. If a GPU library (e.g. CUDA) loads successfully, `gpu` will move data from the CPU to the GPU. As is shown below, this will change the type of something like a regular array to a `CuArray`.

```julia
julia> using Flux, CUDA

julia> m = Dense(10, 5) |> gpu
Dense(10 => 5)      # 55 parameters

julia> x = rand(10) |> gpu
10-element CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}:
 0.066846445
 ⋮
 0.76706964

julia> m(x)
5-element CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}:
 -0.99992573
 ⋮
 -0.547261
```

The analogue `cpu` is also available for moving models and data back off of the GPU.

```julia
julia> x = rand(10) |> gpu
10-element CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}:
 0.8019236
 ⋮
 0.7766742

julia> x |> cpu
10-element Vector{Float32}:
 0.8019236
 ⋮
 0.7766742
```

## Transferring Training Data

In order to train the model using the GPU both model and the training data have to be transferred to GPU memory. Moving the data can be done in two different ways:

1. Iterating over the batches in a [`DataLoader`](@ref) object transferring each one of the training batches at a time to the GPU. This is recommended for large datasets. Done by hand, it might look like this:
   ```julia
   train_loader = Flux.DataLoader((X, Y), batchsize=64, shuffle=true)
   # ... model definition, optimiser setup
   for epoch in 1:epochs
       for (x_cpu, y_cpu) in train_loader
           x = gpu(x_cpu)
           y = gpu(y_cpu)
           grads = gradient(m -> loss(m, x, y), model)
           Flux.update!(opt_state, model, grads[1])
       end
   end
   ```
   Rather than write this out every time, you can just call `gpu(::DataLoader)`:
   ```julia
   gpu_train_loader = Flux.DataLoader((X, Y), batchsize=64, shuffle=true) |> gpu
   # ... model definition, optimiser setup
   for epoch in 1:epochs
       for (x, y) in gpu_train_loader
           grads = gradient(m -> loss(m, x, y), model)
           Flux.update!(opt_state, model, grads[1])
       end
   end
   ```
   This is equivalent to `DataLoader(MLUtils.mapobs(gpu, (X, Y)); keywords...)`.
   Something similar can also be done with [`CUDA.CuIterator`](https://cuda.juliagpu.org/stable/usage/memory/#Batching-iterator), `gpu_train_loader = CUDA.CuIterator(train_loader)`. However, this only works with a limited number of data types: `first(train_loader)` should be a tuple (or `NamedTuple`) of arrays.

2. Transferring all training data to the GPU at once before creating the `DataLoader`. This is usually performed for smaller datasets which are sure to fit in the available GPU memory.
   ```julia
   gpu_train_loader = Flux.DataLoader((X, Y) |> gpu, batchsize = 32)
   # ...
   for epoch in 1:epochs
       for (x, y) in gpu_train_loader
           # ...
   ```
   Here `(X, Y) |> gpu` applies [`gpu`](@ref) to both arrays, as it recurses into structures.

## Saving GPU-Trained Models

After the training process is done, one must always transfer the trained model back to the `cpu` memory scope before serializing or saving to disk. This can be done, as described in the previous section, with:
```julia
model = cpu(model) # or model = model |> cpu
```
and then
```julia
using BSON
# ...
BSON.@save "./path/to/trained_model.bson" model

# in this approach the cpu-transferred model (referenced by the variable `model`)
# only exists inside the `let` statement
let model = cpu(model)
   # ...
   BSON.@save "./path/to/trained_model.bson" model
end

# is equivalent to the above, but uses `key=value` storing directive from BSON.jl
BSON.@save "./path/to/trained_model.bson" model = cpu(model)
```
The reason behind this is that models trained in the GPU but not transferred to the CPU memory scope will expect `CuArray`s as input. In other words, Flux models expect input data coming from the same kind device in which they were trained on.

In controlled scenarios in which the data fed to the loaded models is garanteed to be in the GPU there's no need to transfer them back to CPU memory scope, however in production environments, where artifacts are shared among different processes, equipments or configurations, there is no garantee that the CUDA.jl package will be available for the process performing inference on the model loaded from the disk.


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


More information for conditional use of GPUs in CUDA.jl can be found in its [documentation](https://cuda.juliagpu.org/stable/installation/conditional/#Conditional-use), and information about the specific use of the variable is described in the [Nvidia CUDA blog post](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/).

## Using device objects

As a more convenient syntax, Flux allows the usage of GPU `device` objects which can be used to easily transfer models to GPUs (and defaulting to using the CPU if no GPU backend is available). This syntax has a few advantages including automatic selection of the GPU backend and type stability of data movement. To do this, the [`Flux.get_device`](@ref) function can be used.

`Flux.get_device` first checks for a GPU preference, and if possible returns a device for the preference backend. For instance, consider the following example, where we load the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package to use an NVIDIA GPU (`"CUDA"` is the default preference):

```julia-repl
julia> using Flux, CUDA;

julia> device = Flux.get_device()   # returns handle to an NVIDIA GPU
[ Info: Using backend set in preferences: CUDA.
[ Info: Using backend: CUDA
(::FluxCUDAExt.FluxCUDADevice) (generic function with 1 method)

julia> device.deviceID      # check the id of the GPU
CuDevice(0): NVIDIA GeForce GTX 1650

julia> model = Dense(2 => 3);

julia> model.weight     # the model initially lives in CPU memory
3×2 Matrix{Float32}:
 -0.984794  -0.904345
  0.720379  -0.486398
  0.851011  -0.586942

julia> model = model |> device      # transfer model to the GPU
Dense(2 => 3)       # 9 parameters

julia> model.weight
3×2 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
 -0.984794  -0.904345
  0.720379  -0.486398
  0.851011  -0.586942

```

The device preference can also be set via the [`gpu_backend!`](@ref) function. For instance, below we first set our device preference to `"CPU"`:

```julia-repl
julia> using Flux; Flux.gpu_backend!("CPU")
┌ Info: New GPU backend set: CPU.
└ Restart your Julia session for this change to take effect!
```

Then, after restarting the Julia session, `Flux.get_device` returns a handle to the `"CPU"`:

```julia-repl
julia> using Flux, CUDA;    # even if CUDA is loaded, we'll still get a CPU device

julia> device = Flux.get_device()   # get a CPU device
[ Info: Using backend set in preferences: CPU.
[ Info: Using backend: CPU
(::Flux.FluxCPUDevice) (generic function with 1 method)

julia> model = Dense(2 => 3);

julia> model = model |> device
Dense(2 => 3)       # 9 parameters

julia> model.weight     # no change; model still lives on CPU
3×2 Matrix{Float32}:
 -0.942968   0.856258
  0.440009   0.714106
 -0.419192  -0.471838
```
Clearly, this means that the same code will work for any GPU backend and the CPU. 

If the preference backend isn't available or isn't functional, then [`Flux.get_device`](@ref) looks for a CUDA, AMD or Metal backend, and returns a corresponding device (if the backend is available and functional). Otherwise, a CPU device is returned. For detailed information about how the backend is selected, check the documentation for [`Flux.get_device`](@ref).
