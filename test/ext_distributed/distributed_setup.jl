# Shared backend setup for the `*_distributedtest.jl` child scripts.
#
# The backend is passed through the `FLUX_TEST_DISTRIBUTED_BACKEND` environment
# variable rather than a command-line argument, so that `runtests.jl` can run
# both the "mpi" and the "nccl" passes from a single parent process. Running a
# child script directly therefore requires the variable to be set.

using Flux: DistributedUtils, MPIBackend, NCCLBackend

backend_name = get(ENV, "FLUX_TEST_DISTRIBUTED_BACKEND", nothing)

if backend_name == "mpi"
    import MPI
    const backend_type = MPIBackend
elseif backend_name == "nccl"
    import MPI, NCCL, CUDA
    const backend_type = NCCLBackend
else
    error("`FLUX_TEST_DISTRIBUTED_BACKEND` must be \"mpi\" or \"nccl\"; " *
          "got $(repr(backend_name)).")
end

DistributedUtils.initialize(backend_type)
const backend = DistributedUtils.get_distributed_backend(backend_type)
const rank = DistributedUtils.local_rank(backend)
const tworkers = DistributedUtils.total_workers(backend)
