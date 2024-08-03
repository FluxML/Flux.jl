using Flux, MPI, NCCL, Optimisers, Random, Test, CUDA

const input_args = length(ARGS) == 2 ? ARGS : ("CPU", "mpi")
const backend_type = input_args[2] == "nccl" ? NCCLBackend : MPIBackend
const dev = input_args[1] == "CPU" ? Flux.cpu : Flux.gpu

DistributedUtils.initialize(backend_type)
backend = DistributedUtils.get_distributed_backend(backend_type)

opt = Optimisers.Adam(0.001f0)
ps = (a=zeros(4), b=zeros(4)) |> dev
st_opt = Optimisers.setup(opt, ps)

dopt = DistributedUtils.DistributedOptimizer(backend, opt)
st_dopt = Optimisers.setup(dopt, ps)

@test st_dopt.a.state == st_opt.a.state
@test st_dopt.b.state == st_opt.b.state

@test_nowarn DistributedUtils.synchronize!!(backend, st_dopt)

gs = (a=ones(4), b=ones(4)) |> dev

_, ps_dopt = Optimisers.update(st_dopt, ps, gs)
_, ps_opt = Optimisers.update(st_opt, ps, gs)

@test ps_dopt.a≈ps_opt.a atol=1.0e-5 rtol=1.0e-5
@test ps_dopt.b≈ps_opt.b atol=1.0e-5 rtol=1.0e-5