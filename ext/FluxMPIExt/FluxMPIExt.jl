module FluxMPIExt

using CUDA
# using AMGDPU ### TODO 
using Flux: MPIBackend, NCCLBackend, DistributedUtils,
            AbstractDevice, FluxCUDADevice, FluxAMDGPUDevice, cpu, gpu,
            get_device, MPI_CUDA_AWARE, MPI_ROCM_AWARE
# using LuxDeviceUtils: LuxCUDADevice, LuxAMDGPUDevice
using MPI: MPI


function DistributedUtils.__initialize(
        ::Type{MPIBackend}; cuda_devices=nothing, amdgpu_devices=nothing,
        force_cuda::Bool=false, caller::String="", force_amdgpu::Bool=false) # Undocumented internal kwarg 
    !MPI.Initialized() && MPI.Init()
    DistributedUtils.MPI_Initialized[] = true

    local_rank = MPI.Comm_rank(MPI.COMM_WORLD)

    if cuda_devices !== missing && CUDA.functional()
        if cuda_devices === nothing
            CUDA.device!((local_rank + 1) % length(CUDA.devices()))
        else
            CUDA.device!(cuda_devices[local_rank + 1])
        end
    elseif force_cuda
        error(lazy"CUDA devices are not functional and `force_cuda` is set to `true`. This is caused by backend: $(caller).")
    end

    return
end

DistributedUtils.__get_distributed_backend(::Type{MPIBackend}) = MPIBackend(MPI.COMM_WORLD)

DistributedUtils.local_rank(backend::MPIBackend) = MPI.Comm_rank(backend.comm)

DistributedUtils.total_workers(backend::MPIBackend) = MPI.Comm_size(backend.comm)

# Broadcast
# Union with Function is because of Flux.cpu istypeof Function
# We need CPU in case of non CUDA-aware implementation
function DistributedUtils.__bcast!(
        backend::MPIBackend, sendrecvbuf, dev::Union{AbstractDevice, Function}; root=0)
    MPI.Bcast!(sendrecvbuf, backend.comm; root)
    return sendrecvbuf
end

function DistributedUtils.__bcast!(
        backend::MPIBackend, sendbuf, recvbuf, dev::Union{AbstractDevice, Function}; root=0)
    return DistributedUtils.__bcast!(
        backend, ifelse(DistributedUtils.local_rank(backend) == root, sendbuf, recvbuf),
        dev; root)
end

# if MPI implementation is not CUDA-aware
# we have to move data to CPU first
for (aware, dType) in ((MPI_CUDA_AWARE, FluxCUDADevice), (MPI_ROCM_AWARE, FluxAMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.__bcast!(
                    backend::MPIBackend, sendrecvbuf, dev::$dType; root=0)
                sendrecvbuf_ = sendrecvbuf |> cpu
                DistributedUtils.__bcast!(backend, sendrecvbuf_, cpu; root)
                sendrecvbuf |> gpu
                return sendrecvbuf
            end

            function DistributedUtils.__bcast!(
                    backend::MPIBackend, sendbuf, recvbuf, dev::$dType; root=0)
                sendbuf_ = sendbuf |> cpu
                recvbuf_ = recvbuf |> cpu
                DistributedUtils.__bcast!(backend, sendbuf_, recvbuf_, cpu; root)
                recvbuf |> gpu
                return recvbuf
            end
        end
    end
end


# Allreduce
function DistributedUtils.__allreduce!(
        backend::MPIBackend, sendrecvbuf, op::F, dev::Union{AbstractDevice, Function};) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Allreduce!(sendrecvbuf, mpiop, backend.comm)
    if op === DistributedUtils.avg
        sendrecvbuf ./= DistributedUtils.total_workers(backend)
    end
    return sendrecvbuf
end

function DistributedUtils.__allreduce!(
        backend::MPIBackend, sendbuf, recvbuf, op::F, dev::Union{AbstractDevice, Function};) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Allreduce!(sendbuf, recvbuf, mpiop, backend.comm)
    if op === DistributedUtils.avg
        recvbuf ./= DistributedUtils.total_workers(backend)
    end
    return recvbuf
end

for (aware, dType) in ((MPI_CUDA_AWARE, FluxCUDADevice), (MPI_ROCM_AWARE, FluxAMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.__allreduce!(
                    backend::MPIBackend, sendrecvbuf, op::F, dev::$dType) where {F}
                sendrecvbuf_ = sendrecvbuf |> cpu
                DistributedUtils.__allreduce!(backend, sendrecvbuf_, op, cpu)
                sendrecvbuf |> gpu
                return sendrecvbuf
            end

            function DistributedUtils.__allreduce!(
                    backend::MPIBackend, sendbuf, recvbuf, op::F, dev::$dType) where {F}
                sendbuf_ = sendbuf |> cpu
                recvbuf_ = recvbuf |> cpu
                DistributedUtils.__allreduce!(backend, sendbuf_, recvbuf_, op, cpu)
                recvbuf |> gpu
                return recvbuf
            end
        end
    end
end

# Reduce
function DistributedUtils.__reduce!(backend::MPIBackend, sendrecvbuf, op::F,
        dev::Union{AbstractDevice, Function}; root::Int) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Reduce!(sendrecvbuf, mpiop, backend.comm; root)
    if op === DistributedUtils.avg
        sendrecvbuf ./= DistributedUtils.total_workers(backend)
    end
    return sendrecvbuf
end

function DistributedUtils.__reduce!(backend::MPIBackend, sendbuf, recvbuf, op::F,
        dev::Union{AbstractDevice, Function}; root::Int) where {F}
    mpiop = ifelse(op === DistributedUtils.avg, +, op)
    MPI.Reduce!(sendbuf, recvbuf, mpiop, backend.comm; root)
    if op === DistributedUtils.avg
        recvbuf ./= DistributedUtils.total_workers(backend)
    end
    return recvbuf
end

for (aware, dType) in ((MPI_CUDA_AWARE, FluxCUDADevice), (MPI_ROCM_AWARE, FluxAMDGPUDevice))
    if !aware
        @eval begin
            function DistributedUtils.__reduce!(backend::MPIBackend, sendrecvbuf, op::F,
                    dev::$dType; root::Int) where {F}
                sendrecvbuf_ = sendrecvbuf |> cpu
                DistributedUtils.__reduce!(backend, sendrecvbuf_, op, cpu; root)
                sendrecvbuf |> gpu
                return sendrecvbuf
            end

            function DistributedUtils.__reduce!(backend::MPIBackend, sendbuf, recvbuf,
                    op::F, dev::$dType; root::Int) where {F}
                sendbuf_ = sendbuf |> cpu
                recvbuf_ = recvbuf |> cpu
                DistributedUtils.__reduce!(backend, sendbuf_, recvbuf_, op, cpu; root)
                recvbuf |> gpu
                return recvbuf
            end
        end
    end
end

end