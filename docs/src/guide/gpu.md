# GPU Support

Most work on neural networks involves the use of GPUs, as they can typically perform the required computation much faster.
This page describes how Flux co-operates with various other packages, which talk to GPU hardware.

For those in a hurry, see the [quickstart](@ref man-quickstart) page. Or do `using CUDA` and then call `gpu` on both the model and the data. 

## Basic GPU use: from `Array` to `CuArray`

Julia's GPU packages work with special array types, in place of the built-in `Array`.
The most used is `CuArray` provided by [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), for GPUs made by NVIDIA.
That package provides a function `cu` which converts an ordinary `Array` (living in CPu memory) to a `CuArray` (living in GPU memory).
Functions like `*` and broadcasting specialise so that, when given `CuArray`s, all the computation happens on the GPU:

```julia
W = randn(3, 4)  # some weights, on CPU: 3×4 Array{Float64, 2}
x = randn(4)     # fake data
y = tanh.(W * x) # computation on the CPU

using CUDA

cu(W) isa CuArray{Float32}
(cW, cx) = (W, x) |> cu  # move both to GPU
cy = tanh.(cW * cx)      # computation on the GPU
```

Notice that `cu` doesn't only move arrays, it also recurses into many structures, such as the tuple `(W, x)` above.
(Notice also that it converts Julia's default `Float64` numbers to `Float32`, as this is what most GPUs support efficiently -- it calls itself "opinionated". Flux defaults to `Float32` in all cases.)

To use CUDA with Flux, you can simply use `cu` to move both the model, and the data.
It will create a copy of the Flux model, with all of its parameter arrays moved to the GPU:

```julia
using Pkg; Pkg.add(["CUDA", "cuDNN"])  # do this once

using Flux, CUDA
CUDA.allowscalar(false)  # recommended

model = Dense(W, true, tanh)  # wrap the same matrix W in a Flux layer
model(x) ≈ y                  # same result, still on CPU

c_model = cu(model)  # move all the arrays within model to the GPU
c_model(cx)          # computation on the GPU
```

Notice that you need `using CUDA` (every time) but also `] add cuDNN` (once, when installing packages).
This is a quirk of how these packages are set up.
(The [`cuDNN.jl`](https://github.com/JuliaGPU/CUDA.jl/tree/master/lib/cudnn) sub-package handles operations such as convolutions, called by Flux via [NNlib.jl](https://github.com/FluxML/NNlib.jl).)

Flux's `gradient`, and training functions like `setup`, `update!`, and `train!`, are all equally happy to accept GPU arrays and GPU models, and then perform all computations on the GPU.
It is recommended that you move the model to the GPU before calling `setup`.

```julia
grads = Flux.gradient((f,x) -> sum(abs2, f(x)), model, x)  # on CPU
c_grads = Flux.gradient((f,x) -> sum(abs2, f(x)), c_model, cx)  # same result, all on GPU

c_opt = Flux.setup(Adam(), c_model)  # setup optimiser after moving model to GPU

Flux.update!(c_opt, c_model, c_grads[1])  # mutates c_model but not model
```

To move arrays and other objects back to the CPU, Flux provides a function `cpu`.
This is recommended when saving models, `Flux.state(c_model |> cpu)`, see below.

```julia
cpu(cW) isa Array{Float32, 2}

model2 = cpu(c_model)  # copy model back to CPU
model2(x)
```

!!! compat "Flux ≤ 0.13"
    Old versions of Flux automatically loaded CUDA.jl to provide GPU support. Starting from Flux v0.14, it has to be  loaded separately. Julia's [package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)) allow Flux to automatically load some GPU-specific code when needed.

## Other GPU packages for AMD & Apple

Non-NVIDIA graphics cards are supported by other packages. Each provides its own function which behaves like `cu`.
AMD GPU support provided by [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), on systems with ROCm and MIOpen installed.
This package has a function `roc` which converts `Array` to `ROCArray`:

```julia
using Flux, AMDGPU
AMDGPU.allowscalar(false)

r_model = roc(model)
r_model(roc(x))

Flux.gradient((f,x) -> sum(abs2, f(x)), r_model, roc(x))
```

Experimental support for Apple devices with M-series chips is provided by  [Metal.jl](https://github.com/JuliaGPU/Metal.jl). This has a function [`mtl`](https://metal.juliagpu.org/stable/api/array/#Metal.mtl) which works like `cu`, converting `Array` to `MtlArray`:

```julia
using Flux, Metal
Metal.allowscalar(false)

m_model = mtl(model)
m_y = m_model(mtl(x))

Flux.gradient((f,x) -> sum(abs2, f(x)), m_model, mtl(x))
```

!!! danger "Experimental"
    Metal support in Flux is experimental and many features are not yet available.
    AMD support is improving, but likely to have more rough edges than CUDA.

If you want your model to work with any brand of GPU, or none, then you may not wish to write `cu` everywhere.
One simple way to be generic is, at the top of the file, to un-comment one of several lines which import a package and assign its "adaptor" to the same name:

```julia
using CUDA: cu as device  # after this, `device === cu`
# using AMDGPU: roc as device
# device = identity  # do-nothing, for CPU

using Flux
model = Chain(...) |> device
```

!!! note "Adapt.jl"
    The functions `cu`, `mtl`, `roc` all use [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl), to work within various wrappers.
    The reason they work on Flux models is that `Flux.@layer Layer` defines methods of `Adapt.adapt_structure(to, lay::Layer)`.


## Automatic GPU choice with `gpu` and `gpu_device`

Flux also provides a more automatic way of choosing which GPU (or none) to use. This is the function `gpu`:
* By default it does nothing.
* If the package CUDA is loaded, and `CUDA.functional() === true`, then it behaves like `cu`.
* If the package AMDGPU is loaded,  and `AMDGPU.functional() === true`, then it behaves like `roc`.
* If the package Metal is loaded, and `Metal.functional() === true`, then it behaves like `mtl`.
* If two differnet GPU packages are loaded, the first one takes priority.

For the most part, this means that a script which says `model |> gpu` and `data |> gpu` will just work.
It should always run, and if a GPU package is loaded (and finds the correct hardware) then that will be used.

The function `gpu` uses a lower-level function called [`gpu_device`](@ref) from MLDataDevices.jl,
which checks what to do and then returns some device object. In fact, the entire implementation is just this:

```julia
gpu(x) = gpu_device()(x)
cpu(x) = cpu_device()(x)
```

In case automatic backend selction through `gpu` has an impact in some hot loop of your 
code (although this is rare in practice), it is recommended to first instantiate a device object with `device = gpu_device()`, and then use it to transfer data. 
Finally, setting a backend prefence with [`gpu_backend!`](@ref) gives type stability to the whole pipeline.

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

After the training process is done, we must always transfer the trained model back to the CPU memory before serializing or saving to disk. This can be done with `cpu`:
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


## Data movement across GPU devices

Flux also supports getting handles to specific GPU devices, and transferring models from one GPU device to another GPU device from the same backend. Let's try it out for NVIDIA GPUs. First, we list all the available devices:

```julia-repl
julia> using Flux, CUDA;

julia> CUDA.devices()
CUDA.DeviceIterator() for 3 devices:
0. NVIDIA TITAN RTX
1. NVIDIA TITAN RTX
2. NVIDIA TITAN RTX
```

Then, let's select the device with id `0`:

```julia-repl
julia> device0 = gpu_device(1)
(::CUDADevice{CuDevice}) (generic function with 4 methods)

julia> device0.device
CuDevice(0): NVIDIA TITAN RTX
```
Notice that indexing starts from `0` in the `CUDA.devices()` output, but `gpu_device!` expects the device id starting from `1`.

Then, let's move a simple dense layer to the GPU represented by `device0`:

```julia-repl
julia> dense_model = Dense(2 => 3)
Dense(2 => 3)       # 9 parameters

julia> dense_model = dense_model |> device0;

julia> dense_model.weight
3×2 CuArray{Float32, 2, CUDA.DeviceMemory}:
 -0.142062  -0.131455
 -0.828134  -1.06552
  0.608595  -1.05375

julia> CUDA.device(dense_model.weight)      # check the GPU to which dense_model is attached
CuDevice(0): NVIDIA TITAN RTX
```

Next, we'll get a handle to the device with id `1`, and move `dense_model` to that device:

```julia-repl
julia> device1 = gpu_device(2)
(::CUDADevice{CuDevice}) (generic function with 4 methods)

julia> dense_model = dense_model |> device1;    # don't directly print the model; see warning below

julia> CUDA.device(dense_model.weight)
CuDevice(1): NVIDIA TITAN RTX
```

Due to a limitation in `Metal.jl`, currently this kind of data movement across devices is only supported for `CUDA` and `AMDGPU` backends.


## Distributed data parallel training

!!! danger "Experimental"

    Distributed support is experimental and could change in the future.


Flux supports now distributed data parallel training with `DistributedUtils` module.
If you want to run your code on multiple GPUs, you have to install `MPI.jl` (see [docs](https://juliaparallel.org/MPI.jl/stable/usage/) for more info).

```julia-repl
julia> using MPI

julia> MPI.install_mpiexecjl()
```

Now you can run your code with `mpiexecjl --project=. -n <np> julia <filename>.jl` from CLI.

You can use either the `MPIBackend` or `NCCLBackend`, the latter only if also `NCCL.jl` is loaded. First, initialize a backend with `DistributedUtils.initialize`, e.g.

```julia-repl
julia> using Flux, MPI, NCCL, CUDA

julia> CUDA.allowscalar(false)

julia> DistributedUtils.initialize(NCCLBackend)

julia> backend = DistributedUtils.get_distributed_backend(NCCLBackend)
NCCLBackend{Communicator, MPIBackend{MPI.Comm}}(Communicator(Ptr{NCCL.LibNCCL.ncclComm} @0x000000000607a660), MPIBackend{MPI.Comm}(MPI.Comm(1140850688)))
```

Pass your model, as well as any data to GPU device.
```julia-repl
julia> model = Chain(Dense(1 => 256, tanh), Dense(256 => 1)) |> gpu
Chain(
  Dense(1 => 256, tanh),                # 512 parameters
  Dense(256 => 1),                      # 257 parameters
)                   # Total: 4 arrays, 769 parameters, 744 bytes.

julia> x = rand(Float32, 1, 16) |> gpu
1×16 CUDA.CuArray{Float32, 2, CUDA.DeviceMemory}:
 0.239324  0.331029  0.924996  0.55593  0.853093  0.874513  0.810269  0.935858  0.477176  0.564591  0.678907  0.729682  0.96809  0.115833  0.66191  0.75822

julia> y = x .^ 3
1×16 CUDA.CuArray{Float32, 2, CUDA.DeviceMemory}:
 0.0137076  0.0362744  0.791443  0.171815  0.620854  0.668804  0.53197  0.819654  0.108651  0.179971  0.312918  0.388508  0.907292  0.00155418  0.29  0.435899
```

In this case, we are training on a total of `16 * number of processes` samples. You can also use `DistributedUtils.DistributedDataContainer` to split the data uniformly across processes (or do it manually).

```julia-repl
julia> data = DistributedUtils.DistributedDataContainer(backend, x)
Flux.DistributedUtils.DistributedDataContainer(Float32[0.23932439 0.33102947 … 0.66191036 0.75822026], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
```

You have to wrap your model in `DistributedUtils.FluxDistributedModel` and synchronize it (broadcast accross all processes):
```julia-repl
julia> model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0)
Chain(
  Dense(1 => 256, tanh),                # 512 parameters

  Dense(256 => 1),                      # 257 parameters
)                   # Total: 4 arrays, 769 parameters, 744 bytes.
```

Time to set up an optimizer by using `DistributedUtils.DistributedOptimizer` and synchronize it as well.
```julia-repl
julia> using Optimisers

julia> opt = DistributedUtils.DistributedOptimizer(backend, Optimisers.Adam(0.001f0))
DistributedOptimizer{MPIBackend{Comm}}(MPIBackend{Comm}(Comm(1140850688)), Adam(0.001, (0.9, 0.999), 1.0e-8))

julia> st_opt = Optimisers.setup(opt, model)
(layers = ((weight = Leaf(DistributedOptimizer{MPIBackend{Comm}}(MPIBackend{Comm}(Comm(1140850688)), Adam(0.001, (0.9, 0.999), 1.0e-8)), (Float32[0.0; 0.0; … ; 0.0; 0.0;;], Float32[0.0; 0.0; … ; 0.0; 0.0;;], (0.9, 0.999))), bias = Leaf(DistributedOptimizer{MPIBackend{Comm}}(MPIBackend{Comm}(Comm(1140850688)), Adam(0.001, (0.9, 0.999), 1.0e-8)), (Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (0.9, 0.999))), σ = ()), (weight = Leaf(DistributedOptimizer{MPIBackend{Comm}}(MPIBackend{Comm}(Comm(1140850688)), Adam(0.001, (0.9, 0.999), 1.0e-8)), (Float32[0.0 0.0 … 0.0 0.0], Float32[0.0 0.0 … 0.0 0.0], (0.9, 0.999))), bias = Leaf(DistributedOptimizer{MPIBackend{Comm}}(MPIBackend{Comm}(Comm(1140850688)), Adam(0.001, (0.9, 0.999), 1.0e-8)), (Float32[0.0], Float32[0.0], (0.9, 0.999))), σ = ())),)

julia> st_opt = DistributedUtils.synchronize!!(backend, st_opt; root=0)
(layers = ((weight = Leaf(DistributedOptimizer{MPIBackend{Comm}}(MPIBackend{Comm}(Comm(1140850688)), Adam(0.001, (0.9, 0.999), 1.0e-8)), (Float32[0.0; 0.0; … ; 0.0; 0.0;;], Float32[0.0; 0.0; … ; 0.0; 0.0;;], (0.9, 0.999))), bias = Leaf(DistributedOptimizer{MPIBackend{Comm}}(MPIBackend{Comm}(Comm(1140850688)), Adam(0.001, (0.9, 0.999), 1.0e-8)), (Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (0.9, 0.999))), σ = ()), (weight = Leaf(DistributedOptimizer{MPIBackend{Comm}}(MPIBackend{Comm}(Comm(1140850688)), Adam(0.001, (0.9, 0.999), 1.0e-8)), (Float32[0.0 0.0 … 0.0 0.0], Float32[0.0 0.0 … 0.0 0.0], (0.9, 0.999))), bias = Leaf(DistributedOptimizer{MPIBackend{Comm}}(MPIBackend{Comm}(Comm(1140850688)), Adam(0.001, (0.9, 0.999), 1.0e-8)), (Float32[0.0], Float32[0.0], (0.9, 0.999))), σ = ())),)
```

Now you can define loss and train the model.
```julia-repl
julia> loss(model) = mean((model(x) .- y).^2)
loss (generic function with 1 method)

julia> for epoch in 1:100
           global model, st_opt
           l, grad = Zygote.withgradient(loss, model)
           println("Epoch $epoch: Loss $l")
           st_opt, model = Optimisers.update(st_opt, model, grad[1])
         end
Epoch 1: Loss 0.011638729
Epoch 2: Loss 0.0116432225
Epoch 3: Loss 0.012763695
...
```

Remember that in order to run it on multiple GPUs you have to run from CLI `mpiexecjl --project=. -n <np> julia <filename>.jl`,
where  `<np>` is the number of processes that you want to use. The number of processes usually corresponds to the number of gpus.

By default `MPI.jl` MPI installation is CUDA-unaware so if you want to run it in CUDA-aware mode, read more [here](https://juliaparallel.org/MPI.jl/stable/usage/#CUDA-aware-MPI-support) on custom installation and rebuilding `MPI.jl`.
Then test if your MPI is CUDA-aware by
```julia-repl
julia> import Pkg
julia> Pkg.test("MPI"; test_args=["--backend=CUDA"])
```

If it is, set your local preference as below
```julia-repl
julia> using Preferences
julia> set_preferences!("Flux", "FluxDistributedMPICUDAAware" => true)
```

!!! warning "Known shortcomings"

    We don't run CUDA-aware tests so you're running it at own risk.


## Checking GPU Availability

By default, Flux will run the checks on your system to see if it can support GPU functionality. You can check if Flux identified a valid GPU setup by typing the following:

```julia-repl
julia> using CUDA

julia> CUDA.functional()
true
```

For AMD GPU:

```julia-repl
julia> using AMDGPU

julia> AMDGPU.functional()
true

julia> AMDGPU.functional(:MIOpen)
true
```

For Metal GPU:

```julia-repl
julia> using Metal

julia> Metal.functional()
true
```
