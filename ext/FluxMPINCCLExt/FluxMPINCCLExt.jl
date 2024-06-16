module FluxMPINCCLExt

using Flux: MPIBackend, NCCLBackend, DistributedUtils, FluxCUDADevice, FluxAMDGPUDevice, AbstractDevice
using MPI: MPI
using NCCL: NCCL
using Setfield: @set!
using CUDA

function DistributedUtils.__initialize(
        ::Type{NCCLBackend}; cuda_devices=nothing, amdgpu_devices=missing)
    @assert amdgpu_devices===missing "`AMDGPU` is not supported by `NCCL`."
    DistributedUtils.__initialize(
        MPIBackend; cuda_devices, force_cuda=true, caller="NCCLBackend", amdgpu_devices)
    DistributedUtils.NCCL_Initialized[] = true
    return
end

function DistributedUtils.__get_distributed_backend(::Type{NCCLBackend})
    unique_id = NCCL.UniqueID()  # Generate on all ranks to know the type
    mpi_backend = DistributedUtils.__get_distributed_backend(MPIBackend)
    buf = [unique_id.internal...]
    DistributedUtils.bcast!(mpi_backend, buf; root=0)
    @set! unique_id.internal = Tuple(buf)

    nranks = DistributedUtils.total_workers(mpi_backend)
    rank = DistributedUtils.local_rank(mpi_backend)

    return NCCLBackend(NCCL.Communicator(nranks, rank; unique_id), mpi_backend)
end

DistributedUtils.local_rank(backend::NCCLBackend) = NCCL.rank(backend.comm)

DistributedUtils.total_workers(backend::NCCLBackend) = NCCL.size(backend.comm)

# For non-CUDA Arrays, fallback to MPI
# Broadcast
function DistributedUtils.__bcast!(
        backend::NCCLBackend, sendrecvbuf::CuArray, ::FluxCUDADevice; root=0)
    NCCL.Broadcast!(sendrecvbuf, backend.comm; root)
    return sendrecvbuf
end

function DistributedUtils.__bcast!(
        backend::NCCLBackend, sendrecvbuf, dev::AbstractDevice; root=0)
    return DistributedUtils.__bcast!(backend.mpi_backend, sendrecvbuf, dev; root)
end

function DistributedUtils.__bcast!(
        backend::NCCLBackend, sendbuf, recvbuf, ::FluxCUDADevice; root=0)
    NCCL.Broadcast!(sendbuf, recvbuf, backend.comm; root)
    return recvbuf
end

function DistributedUtils.__bcast!(
        backend::NCCLBackend, sendbuf, recvbuf, dev::AbstractDevice; root=0)
    return DistributedUtils.__bcast!(backend.mpi_backend, sendbuf, recvbuf, dev; root)
end

# Allreduce
function DistributedUtils.__allreduce!(
        backend::NCCLBackend, sendrecvbuf::CuArray, op::F, dev::FluxCUDADevice) where {F}
    op = ifelse(op === DistributedUtils.avg, NCCL.avg, op)
    NCCL.Allreduce!(sendrecvbuf, op, backend.comm)
    return sendrecvbuf
end

function DistributedUtils.__allreduce!(
        backend::NCCLBackend, sendrecvbuf, op::F, dev::AbstractDevice) where {F}
    return DistributedUtils.__allreduce!(backend.mpi_backend, sendrecvbuf, op, dev)
end

function DistributedUtils.__allreduce!(
        backend::NCCLBackend, sendbuf, recvbuf, op::F, ::FluxCUDADevice) where {F}
    op = ifelse(op === DistributedUtils.avg, NCCL.avg, op)
    NCCL.Allreduce!(sendbuf, recvbuf, op, backend.comm)
    return recvbuf
end

function DistributedUtils.__allreduce!(
        backend::NCCLBackend, sendbuf, recvbuf, op::F, dev::AbstractDevice) where {F}
    return DistributedUtils.__allreduce!(backend.mpi_backend, sendbuf, recvbuf, op, dev)
end

# Reduce
function DistributedUtils.__reduce!(
        backend::NCCLBackend, sendrecvbuf, op::F, ::FluxCUDADevice; root::Int) where {F}
    op = ifelse(op === DistributedUtils.avg, NCCL.avg, op)
    NCCL.Reduce!(sendrecvbuf, op, backend.comm; root)
    return sendrecvbuf
end

function DistributedUtils.__reduce!(backend::NCCLBackend, sendrecvbuf, op::F,
        dev::AbstractDevice; root::Int) where {F}
    return DistributedUtils.__reduce!(backend.mpi_backend, sendrecvbuf, op, dev; root)
end

function DistributedUtils.__reduce!(
        backend::NCCLBackend, sendbuf, recvbuf, op::F, ::FluxCUDADevice; root::Int) where {F}
    op = ifelse(op === DistributedUtils.avg, NCCL.avg, op)
    NCCL.Reduce!(sendbuf, recvbuf, op, backend.comm; root)
    return recvbuf
end

function DistributedUtils.__reduce!(backend::NCCLBackend, sendbuf, recvbuf, op::F,
        dev::AbstractDevice; root::Int) where {F}
    return DistributedUtils.__reduce!(backend.mpi_backend, sendbuf, recvbuf, op, dev; root)
end

end