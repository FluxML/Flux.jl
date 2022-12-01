# GPU Support

NVIDIA GPU support should work out of the box on systems with CUDA and CUDNN installed. For more details see the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) readme.

## Checking GPU Availability

By default, Flux will run the checks on your system to see if it can support GPU functionality. You can check if Flux identified a valid GPU setup by typing the following:

```julia
julia> using CUDA

julia> CUDA.functional()
true
```

## GPU Usage

Support for array operations on other hardware backends, like GPUs, is provided by external packages like [CUDA](https://github.com/JuliaGPU/CUDA.jl). Flux is agnostic to array types, so we simply need to move model weights and data to the GPU and Flux will handle it.

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
d(cu(rand(10)))
```

As a convenience, Flux provides the `gpu` function to convert models and data to the GPU if one is available. By default, it'll do nothing. So, you can safely call `gpu` on some data or model (as shown below), and the code will not error, regardless of whether the GPU is available or not. If the GPU library (CUDA.jl) loads successfully, `gpu` will move data from the CPU to the GPU. As is shown below, this will change the type of something like a regular array to a `CuArray`.

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

```@docs
cpu
gpu
```

## Common GPU Workflows

Some of the common workflows involving the use of GPUs are presented below.

### Transferring Training Data

In order to train the model using the GPU both model and the training data have to be transferred to GPU memory. This process can be done with the `gpu` function in two different ways:

1. Iterating over the batches in a [DataLoader](@ref) object transferring each one of the training batches at a time to the GPU. 
   ```julia
   train_loader = Flux.DataLoader((xtrain, ytrain), batchsize = 64, shuffle = true)
   # ... model, optimizer and loss definitions
   for epoch in 1:nepochs
       for (xtrain_batch, ytrain_batch) in train_loader
           x, y = gpu(xtrain_batch), gpu(ytrain_batch)
           gradients = gradient(() -> loss(x, y), parameters)
           Flux.Optimise.update!(optimizer, parameters, gradients)
       end
   end
   ```

2. Transferring all training data to the GPU at once before creating the [DataLoader](@ref) object. This is usually performed for smaller datasets which are sure to fit in the available GPU memory. Some possibilities are:
   ```julia
   gpu_train_loader = Flux.DataLoader((xtrain |> gpu, ytrain |> gpu), batchsize = 32)
   ```
   ```julia
   gpu_train_loader = Flux.DataLoader((xtrain, ytrain) |> gpu, batchsize = 32)
   ```
   Note that both `gpu` and `cpu` are smart enough to recurse through tuples and namedtuples. Another possibility is to use [`MLUtils.mapsobs`](https://juliaml.github.io/MLUtils.jl/dev/api/#MLUtils.mapobs) to push the data movement invocation into the background thread:
   ```julia
   using MLUtils: mapobs
   # ...
   gpu_train_loader = Flux.DataLoader(mapobs(gpu, (xtrain, ytrain)), batchsize = 16)
   ```

3. Wrapping the `DataLoader` in [`CUDA.CuIterator`](https://cuda.juliagpu.org/stable/usage/memory/#Batching-iterator) to efficiently move data to GPU on demand:
   ```julia
   using CUDA: CuIterator
   train_loader = Flux.DataLoader((xtrain, ytrain), batchsize = 64, shuffle = true)
   # ... model, optimizer and loss definitions
   for epoch in 1:nepochs
       for (xtrain_batch, ytrain_batch) in CuIterator(train_loader)
          # ...
       end
   end
   ```

   Note that this works with a limited number of data types. If `iterate(train_loader)` returns anything other than arrays, approach 1 or 2 is preferred.

### Saving GPU-Trained Models

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


## CUDA Reference

Arrays and iterators:

```@docs
CUDA.cu
CUDA.CuIterator
```

Device settings:

```@docs
CUDA.allowscalar
CUDA.functional
CUDA.device
CUDA.device!
```

For benchmarking:

```@docs
CUDA.@time
CUDA.@sync
```
