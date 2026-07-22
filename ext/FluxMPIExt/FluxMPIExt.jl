module FluxMPIExt

using Flux: MPIBackend, NCCLBackend, DistributedUtils,
            MPI_CUDA_AWARE, MPI_ROCM_AWARE
using MPI: MPI
using MLDataDevices: AbstractDevice, CUDADevice, AMDGPUDevice, functional, set_device!, cpu_device


function DistributedUtils.__initialize(
        ::Type{MPIBackend}; cuda_devices=nothing, amdgpu_devices=nothing,
        force_cuda::Bool=false, caller::String="",
        # Undocumented internal kwargs
        force_amdgpu::Bool=false, force::Bool=false)

    if !force && !MPI.Initialized()
        if haskey(ENV, "PMIX_RANK")
            error(
                "MPI backend initialization failed: Found PMIX_RANK in " *
                "environment, but Julia's default MPICH_jll uses PMI2. " *
                "The runtime environment uses an unsupported PMI version " *
                "(PMIx), which will cause a hard abort in MPI.Init(). " *
                "Please launch your job with a PMI2-compatible launcher " *
                "(e.g., `srun --mpi=pmi2 ...`) or rebuild MPI.jl with " *
                "your system MPI. To bypass this check, pass `force=true` " *
                "to `initialize(MPIBackend; force=true)`."
            )
        elseif haskey(ENV, "OMPI_COMM_WORLD_RANK")
            error(
                "MPI backend initialization failed: Found " *
                "OMPI_COMM_WORLD_RANK in environment, but Julia's " *
                "default MPICH_jll expects MPICH. You appear to have " *
                "launched with OpenMPI's mpirun, but loaded MPICH_jll. " *
                "Please launch with an MPICH-compatible launcher or " *
                "rebuild MPI.jl with OpenMPI. To bypass this check, pass " *
                "`force=true` to `initialize(MPIBackend; force=true)`."
            )
        elseif haskey(ENV, "SLURM_JOB_ID") &&
               !any(startswith(k, "PMI2") for k in keys(ENV))
            @warn(
                "No PMI2 environment variables found inside a Slurm " *
                "allocation. Julia's default MPICH_jll uses PMI2. If " *
                "`MPI.Init()` aborts or hangs, ensure you launched " *
                "with `srun --mpi=pmi2`."
            )
        end
    end

    !MPI.Initialized() && MPI.Init()
    DistributedUtils.MPI_Initialized[] = true

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)

    if cuda_devices !== missing && functional(CUDADevice)
        if cuda_devices === nothing
            set_device!(CUDADevice, nothing, local_rank + 1)
        else
            set_device!(CUDADevice, cuda_devices[local_rank + 1])
        end
    elseif force_cuda
        error(lazy"CUDA devices are not functional and `force_cuda` is set to `true`. This is caused by backend: $(caller).")
    end

    if amdgpu_devices !== missing && functional(AMDGPUDevice)
        if amdgpu_devices === nothing
            set_device!(AMDGPUDevice, nothing, local_rank + 1)
        else
            set_device!(AMDGPUDevice, amdgpu_devices[local_rank + 1])
        end
    elseif force_amdgpu
        error(lazy"AMDGPU devices are not functional (or `LuxAMDGPU.jl` not loaded) and `force_amdgpu` is set to `true`. This is caused by backend: $(caller).")
    end

    return
end

DistributedUtils.__get_distributed_backend(::Type{MPIBackend}) = MPIBackend(MPI.COMM_WORLD)

DistributedUtils.local_rank(backend::MPIBackend) = MPI.Comm_rank(backend.comm)

DistributedUtils.total_workers(backend::MPIBackend) = MPI.Comm_size(backend.comm)

# Broadcast
# We need CPU in case of non CUDA-aware implementation
function DistributedUtils.__bcast!(
        backend::MPIBackend, sendrecvbuf, dev::AbstractDevice; root=0)
    MPI.Bcast!(sendrecvbuf, backend.comm; root)
    return sendrecvbuf
end

function DistributedUtils.__bcast!(
        backend::MPIBackend, sendbuf, recvbuf, dev::AbstractDevice; root=0)
    return DistributedUtils.__bcast!(
        backend, ifelse(DistributedUtils.local_rank(backend) == root, sendbuf, recvbuf),
        dev; root)
end

# if MPI implementation is not CUDA-aware
# we have to move data to CPU first
for (aware, dType) in ((MPI_CUDA_AWARE, CUDADevice), (MPI_ROCM_AWARE, AMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.__bcast!(
                    backend::MPIBackend, sendrecvbuf, dev::$dType; root=0)
                cdev = cpu_device()
                sendrecvbuf_ = sendrecvbuf |> cdev
                DistributedUtils.__bcast!(backend, sendrecvbuf_, cdev; root)
                copyto!(sendrecvbuf, sendrecvbuf_)
                return
            end

            function DistributedUtils.__bcast!(
                    backend::MPIBackend, sendbuf, recvbuf, dev::$dType; root=0)
                cdev = cpu_device()
                sendbuf_ = sendbuf |> cdev
                recvbuf_ = recvbuf |> cdev
                DistributedUtils.__bcast!(backend, sendbuf_, recvbuf_, cdev; root)
                copyto!(recvbuf, recvbuf_)
                return
            end
        end
    end
end


# Allreduce
function DistributedUtils.__allreduce!(
        backend::MPIBackend, sendrecvbuf, op::F, ::AbstractDevice) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Allreduce!(sendrecvbuf, mpiop, backend.comm)
    if op === DistributedUtils.avg
        sendrecvbuf ./= DistributedUtils.total_workers(backend)
    end
    return sendrecvbuf
end

function DistributedUtils.__allreduce!(
        backend::MPIBackend, sendbuf, recvbuf, op::F, ::AbstractDevice) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Allreduce!(sendbuf, recvbuf, mpiop, backend.comm)
    if op === DistributedUtils.avg
        recvbuf ./= DistributedUtils.total_workers(backend)
    end
    return recvbuf
end

for (aware, dType) in ((MPI_CUDA_AWARE, CUDADevice), (MPI_ROCM_AWARE, AMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.__allreduce!(
                    backend::MPIBackend, sendrecvbuf, op::F, dev::$dType) where {F}
                cdev = cpu_device()
                sendrecvbuf_ = sendrecvbuf |> cdev
                DistributedUtils.__allreduce!(backend, sendrecvbuf_, op, cdev)
                copyto!(sendrecvbuf, sendrecvbuf_)
                return
            end

            function DistributedUtils.__allreduce!(
                    backend::MPIBackend, sendbuf, recvbuf, op::F, dev::$dType) where {F}
                cdev = cpu_device()
                sendbuf_ = sendbuf |> cdev
                recvbuf_ = recvbuf |> cdev
                DistributedUtils.__allreduce!(backend, sendbuf_, recvbuf_, op, cdev)
                copyto!(recvbuf, recvbuf_)
                return
            end
        end
    end
end

# Reduce
function DistributedUtils.__reduce!(backend::MPIBackend, sendrecvbuf, op::F,
        dev::AbstractDevice; root::Int) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Reduce!(sendrecvbuf, mpiop, backend.comm; root)
    if op === DistributedUtils.avg
        sendrecvbuf ./= DistributedUtils.total_workers(backend)
    end
    return sendrecvbuf
end

function DistributedUtils.__reduce!(backend::MPIBackend, sendbuf, recvbuf, op::F,
        dev::AbstractDevice; root::Int) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Reduce!(sendbuf, recvbuf, mpiop, backend.comm; root)
    if op === DistributedUtils.avg
        recvbuf ./= DistributedUtils.total_workers(backend)
    end
    return recvbuf
end

for (aware, dType) in ((MPI_CUDA_AWARE, CUDADevice), (MPI_ROCM_AWARE, AMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.__reduce!(backend::MPIBackend, sendrecvbuf, op::F,
                    dev::$dType; root::Int) where {F}
                cdev = cpu_device()
                sendrecvbuf_ = sendrecvbuf |> cdev
                DistributedUtils.__reduce!(backend, sendrecvbuf_, op, cdev; root)
                copyto!(sendrecvbuf, sendrecvbuf_)
                return
            end

            function DistributedUtils.__reduce!(backend::MPIBackend, sendbuf, recvbuf,
                    op::F, dev::$dType; root::Int) where {F}
                cdev = cpu_device()
                sendbuf_ = sendbuf |> cdev
                recvbuf_ = recvbuf |> cdev
                DistributedUtils.__reduce!(backend, sendbuf_, recvbuf_, op, cdev; root)
                copyto!(recvbuf, recvbuf_)
                return
            end
        end
    end
end

end