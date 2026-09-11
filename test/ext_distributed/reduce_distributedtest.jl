using Test

include(joinpath(@__DIR__, "distributed_setup.jl"))

sendrecvbuf = fill(rank+1,4)

DistributedUtils.reduce!(backend, sendrecvbuf, +)

if rank == 0
    @test all(sendrecvbuf .== sum(1:tworkers))
end
