const input_args = length(ARGS) == 2 ? ARGS : ("CPU", "mpi")
const backend_type = input_args[2] == "nccl" ? NCCLBackend : MPIBackend
const dev = input_args[1] == "CPU" ? Flux.cpu : Flux.gpu

rng = Xoshiro(1234)

DistributedUtils.initialize(backend_type)
backend = DistributedUtils.get_distributed_backend(backend_type)

data = randn(rng, Float32, 10)
dcontainer = DistributedUtils.DistributedDataContainer(backend, data)

rank = DistributedUtils.local_rank(backend)
tworkers = DistributedUtils.total_workers(backend)

if rank != tworkers - 1
    @test length(dcontainer) == ceil(length(data) / tworkers)
else
    @test length(dcontainer) ==
          length(data) - (tworkers - 1) * ceil(length(data) / tworkers)
end

dsum = sum(Base.Fix1(MLUtils.getobs, dcontainer), 1:MLUtils.numobs(dcontainer))
@test DistributedUtils.allreduce!(backend, [dsum], +)[1] ≈ sum(data)