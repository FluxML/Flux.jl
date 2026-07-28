# Distributed Training

Flux provides native support for Distributed Data Parallel (DDP) training through the `DistributedUtils` module. DDP allows you to train a single model across multiple GPUs or CPUs by running identical copies of the model on each device. During training, each device computes gradients on a different chunk of data (sharding), and the gradients are synchronized (averaged) across all devices before the optimizer updates the model parameters.

## Initialization

DDP uses MPI (and optionally NCCL for GPUs) under the hood. You must initialize the distributed backend at the start of your script. If you have CUDA available, use `NCCLBackend`; otherwise, use `MPIBackend`.

```julia
using Flux
using Flux: DistributedUtils
using MPI
using Optimisers
using MLUtils
using Zygote

# Determine if we should use GPU (NCCL) or CPU (MPI)
const USE_CUDA = try
    using CUDA, NCCL
    CUDA.functional()
catch
    false
end

if USE_CUDA
    DistributedUtils.initialize(DistributedUtils.NCCLBackend)
    backend = DistributedUtils.get_distributed_backend(DistributedUtils.NCCLBackend)
else
    DistributedUtils.initialize(DistributedUtils.MPIBackend)
    backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)
end

rank = DistributedUtils.local_rank(backend)
world = DistributedUtils.total_workers(backend)
```

## Sharding Data

To ensure each process sees a different slice of the dataset, wrap your dataset in a `DistributedDataContainer`. When iterating, it will automatically yield a distinct subset of the data based on the process's `rank`.

```julia
using Random

# Synthetic validation dataset (same for all ranks before sharding)
Random.seed!(42)
val_dataset = (rand(Float32, 10, 200), rand(Float32, 2, 200))
ddp_val_data = DistributedUtils.DistributedDataContainer(backend, val_dataset)
val_data_loader = DataLoader(ddp_val_data, batchsize=32, shuffle=false)

# Synthetic training dataset
Random.seed!(42) # Use a fixed global seed so all ranks generate the identical global dataset before sharding
dataset = (rand(Float32, 10, 1000), rand(Float32, 2, 1000))

# Shard the dataset across all processes
ddp_data = DistributedUtils.DistributedDataContainer(backend, dataset)

# MLUtils DataLoader handles batching as usual
data_loader = DataLoader(ddp_data, batchsize=32, shuffle=true)
```

## Model and Optimizer Setup

Before training, you must ensure that all processes start with the exact same initial weights and optimizer state. We use `synchronize!!` to broadcast the state from `rank = 0` to all other ranks.

```julia
model = Chain(Dense(10 => 32, relu), Dense(32 => 2))

if USE_CUDA
    model = fmap(gpu, model)
end

# 1. Sync Model
# We must wrap the model to correctly identify all parameter arrays
ddp_model = DistributedUtils.FluxDistributedModel(model)
model = DistributedUtils.synchronize!!(backend, ddp_model; root=0)

# 2. Setup Optimizer Rule
# Wrap your base rule in `DistributedOptimizer` to enable automatic gradient averaging
dist_opt = DistributedUtils.DistributedOptimizer(backend, Optimisers.Adam(0.01))

# 3. Setup and Sync Optimizer State
opt_state = Optimisers.setup(dist_opt, model)
opt_state = DistributedUtils.synchronize!!(backend, opt_state; root=0)
```

## Training Loop

The training loop proceeds identically to standard single-process training. Because we wrapped our optimizer rule in `DistributedOptimizer`, calling `Optimisers.update` will automatically trigger an `allreduce` operation to average the gradients across all processes.

```julia
epochs = 5
for epoch in 1:epochs
    total_loss = 0.0f0
    batches = 0
    
    for (x, y) in data_loader
        if USE_CUDA
            x, y = gpu(x), gpu(y)
        end
        
        l, gs = Zygote.withgradient(model) do m
            Flux.Losses.mse(m(x), y)
        end
        gs = gs[1]
        
        # Apply gradients (automatically averages across ranks)
        opt_state, model = Optimisers.update(opt_state, model, gs)
        
        total_loss += l
        batches += 1
    end
    
    avg_loss = total_loss / batches
    
    # (Optional) Average the printed loss across all processes for logging
    global_loss = DistributedUtils.allreduce!(backend, [avg_loss], +)[1] / world
    
    if rank == 0
        println("Epoch $epoch | Global Train Loss: $global_loss")
    end

    # Validation Loop
    val_total_loss = 0.0f0
    val_batches = 0
    
    for (x, y) in val_data_loader
        if USE_CUDA
            x, y = gpu(x), gpu(y)
        end
        
        # Compute loss without tracking gradients
        l = Flux.Losses.mse(model(x), y)
        
        val_total_loss += l
        val_batches += 1
    end
    
    avg_val_loss = val_total_loss / val_batches
    global_val_loss = DistributedUtils.allreduce!(backend, [avg_val_loss], +)[1] / world
    
    if rank == 0
        println("Epoch $epoch | Global Val Loss: $global_val_loss")
    end
end
```

During validation, `allreduce!` is necessary because each rank evaluates only a partition (shard) of the validation dataset. To compute the true global validation metric (such as average loss or accuracy), we must aggregate the locally computed metrics across all distributed partitions. By averaging the validation metrics via `allreduce!` before logging, we ensure that Rank 0 reports an accurate and comprehensive evaluation of the model's performance on the entire validation set. Always ensure validation metrics are only logged by Rank 0 to avoid cluttering standard output.

## Advanced: Conditional Graphs and Unused Parameters

If your model has branches and a particular parameter is completely unused during the forward/backward pass (for instance, a multi-task network where only certain heads are active for a specific batch), `Zygote` will return `nothing` for that parameter's gradient. 

During the DDP `allreduce` step, these `nothing` values will cause a type mismatch or crash if other processes computed a gradient for that parameter. 

To safely handle conditional graphs, use `resolve_unused_parameters!!` immediately after `withgradient`. This replaces any `nothing` gradients with zero-filled arrays of the correct shape and type:

```julia
        l, gs = Zygote.withgradient(model) do m
            y_hat = m(x, rank == 0) # Only rank 0 uses the first branch
            Flux.Losses.mse(y_hat, y)
        end
        gs = gs[1]
        
        # Replace `nothing` with zeros to prevent DDP deadlocks
        gs = DistributedUtils.resolve_unused_parameters!!(backend, gs, model)
        
        opt_state, model = Optimisers.update(opt_state, model, gs)
```

## Saving and Checkpointing

In a distributed setting, model saving *must* be restricted to a single worker (typically rank 0) to avoid I/O race conditions and prevent file corruption.

```julia
# Ensure JLD2 is imported (e.g., `using JLD2`) at the top of your file
if rank == 0
    println("Training complete. Saving model checkpoint...")
    model_state = Flux.state(model)
    jldsave("ddp_model_checkpoint.jld2"; model_state)
end
```
