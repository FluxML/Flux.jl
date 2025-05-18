const input_args = length(ARGS) == 2 ? ARGS : ("CPU", "mpi")
const backend_type = input_args[2] == "nccl" ? NCCLBackend : MPIBackend
const dev = input_args[1] == "CPU" ? Flux.cpu : Flux.gpu
const aType = input_args[1] == "CPU" ? Array :
              (input_args[1] == "CUDA" ? CuArray : ROCArray)

DistributedUtils.initialize(backend_type)
backend = DistributedUtils.get_distributed_backend(backend_type)

@test DistributedUtils.initialized(backend_type)

# Should always hold true
rank = DistributedUtils.local_rank(backend)
nworkers = DistributedUtils.total_workers(backend)
@test rank < nworkers

# Test the communication primitives
## broacast!
for arrType in (Array, aType)
    sendbuf = (rank == 0) ? arrType(ones(512)) : arrType(zeros(512))
    recvbuf = arrType(zeros(512))

    DistributedUtils.bcast!(backend, sendbuf, recvbuf; root=0)

    rank != 0 && @test all(recvbuf .== 1)

    sendrecvbuf = (rank == 0) ? arrType(ones(512)) : arrType(zeros(512))
    DistributedUtils.bcast!(backend, sendrecvbuf; root=0)

    @test all(sendrecvbuf .== 1)
end

## reduce!
for arrType in (Array, aType)
    sendbuf = arrType(fill(Float64(rank + 1), 512))
    recvbuf = arrType(zeros(512))

    DistributedUtils.reduce!(backend, sendbuf, recvbuf, +; root=0)

    rank == 0 && @test all(recvbuf .≈ sum(1:nworkers))

    sendbuf .= rank + 1

    DistributedUtils.reduce!(backend, sendbuf, recvbuf, DistributedUtils.avg; root=0)

    rank == 0 && @test all(recvbuf .≈ sum(1:nworkers) / nworkers)

    sendrecvbuf = arrType(fill(Float64(rank + 1), 512))

    DistributedUtils.reduce!(backend, sendrecvbuf, +; root=0)

    rank == 0 && @test all(sendrecvbuf .≈ sum(1:nworkers))

    sendrecvbuf .= rank + 1

    DistributedUtils.reduce!(backend, sendrecvbuf, DistributedUtils.avg; root=0)

    rank == 0 && @test all(sendrecvbuf .≈ sum(1:nworkers) / nworkers)
end

## allreduce!
for arrType in (Array, aType)
    sendbuf = arrType(fill(Float64(rank + 1), 512))
    recvbuf = arrType(zeros(512))

    DistributedUtils.allreduce!(backend, sendbuf, recvbuf, +)

    @test all(recvbuf .≈ sum(1:nworkers))

    sendbuf .= rank + 1

    DistributedUtils.allreduce!(backend, sendbuf, recvbuf, DistributedUtils.avg)

    @test all(recvbuf .≈ sum(1:nworkers) / nworkers)

    sendrecvbuf = arrType(fill(Float64(rank + 1), 512))

    DistributedUtils.allreduce!(backend, sendrecvbuf, +)

    @test all(sendrecvbuf .≈ sum(1:nworkers))

    sendrecvbuf .= rank + 1

    DistributedUtils.allreduce!(backend, sendrecvbuf, DistributedUtils.avg)

    @test all(sendrecvbuf .≈ sum(1:nworkers) / nworkers)
end