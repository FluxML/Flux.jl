using Test
using Flux

include(joinpath(@__DIR__, "distributed_setup.jl"))

const dev = Flux.cpu
const aType = Array

@test DistributedUtils.initialized(backend_type)

# Should always hold true
@test rank < tworkers

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

    rank == 0 && @test all(recvbuf .≈ sum(1:tworkers))

    sendbuf .= rank + 1

    DistributedUtils.reduce!(backend, sendbuf, recvbuf, DistributedUtils.avg; root=0)

    rank == 0 && @test all(recvbuf .≈ sum(1:tworkers) / tworkers)

    sendrecvbuf = arrType(fill(Float64(rank + 1), 512))

    DistributedUtils.reduce!(backend, sendrecvbuf, +; root=0)

    rank == 0 && @test all(sendrecvbuf .≈ sum(1:tworkers))

    sendrecvbuf .= rank + 1

    DistributedUtils.reduce!(backend, sendrecvbuf, DistributedUtils.avg; root=0)

    rank == 0 && @test all(sendrecvbuf .≈ sum(1:tworkers) / tworkers)
end

## allreduce!
for arrType in (Array, aType)
    sendbuf = arrType(fill(Float64(rank + 1), 512))
    recvbuf = arrType(zeros(512))

    DistributedUtils.allreduce!(backend, sendbuf, recvbuf, +)

    @test all(recvbuf .≈ sum(1:tworkers))

    sendbuf .= rank + 1

    DistributedUtils.allreduce!(backend, sendbuf, recvbuf, DistributedUtils.avg)

    @test all(recvbuf .≈ sum(1:tworkers) / tworkers)

    sendrecvbuf = arrType(fill(Float64(rank + 1), 512))

    DistributedUtils.allreduce!(backend, sendrecvbuf, +)

    @test all(sendrecvbuf .≈ sum(1:tworkers))

    sendrecvbuf .= rank + 1

    DistributedUtils.allreduce!(backend, sendrecvbuf, DistributedUtils.avg)

    @test all(sendrecvbuf .≈ sum(1:tworkers) / tworkers)
end