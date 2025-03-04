using Flux
using Test

backend_string = ARGS[1]

if backend_string == "mpi"
    import MPI
    backend_type = MPIBackend
elseif backend_string == "nccl"
    import MPI, NCCL, CUDA
    backend_type = NCCLBackend
else
    error("unsupported backend: $backend_string")
end

DistributedUtils.initialize(backend_type)
backend = DistributedUtils.get_distributed_backend(backend_type)

rank = DistributedUtils.local_rank(backend)
total_workers = DistributedUtils.total_workers(backend)

sendrecvbuf = fill(rank+1,4)

DistributedUtils.reduce!(backend, sendrecvbuf, +)

if rank == 0
    @test all(sendrecvbuf .== sum(1:total_workers))
end
