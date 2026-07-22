using Test
using Random
using MLUtils
using Flux
using Flux: DistributedUtils, MPIBackend, NCCLBackend

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

rng = Xoshiro(1234)

DistributedUtils.initialize(backend_type)
backend = DistributedUtils.get_distributed_backend(backend_type)

rank = DistributedUtils.local_rank(backend)
tworkers = DistributedUtils.total_workers(backend)

# Test 1: Evenly divisible dataset (no padding needed)
data_even = randn(rng, Float32, 4 * tworkers)
dc_even = DistributedUtils.DistributedDataContainer(backend, data_even)

@test length(dc_even) == div(length(data_even), tworkers)

dsum = sum(Base.Fix1(MLUtils.getobs, dc_even), 1:MLUtils.numobs(dc_even))
@test DistributedUtils.allreduce!(backend, [dsum], +)[1] ≈ sum(data_even)

# Test 2: Non-divisible dataset — padding ensures equal partition sizes
data_odd = randn(rng, Float32, 7)
dc_odd = DistributedUtils.DistributedDataContainer(backend, data_odd)

expected_len = Int(ceil(length(data_odd) / tworkers))
@test length(dc_odd) == expected_len

# All original elements must appear across the partitions.
# Padding duplicates elements from the beginning, so the global sum includes those.
pad_count = expected_len * tworkers - length(data_odd)
expected_sum = sum(data_odd) + (pad_count > 0 ? sum(data_odd[1:pad_count]) : 0.0f0)

dsum_odd = sum(Base.Fix1(MLUtils.getobs, dc_odd), 1:MLUtils.numobs(dc_odd))
@test DistributedUtils.allreduce!(backend, [dsum_odd], +)[1] ≈ expected_sum

# Test 3: Original dataset — preserve the original 10-element test case
data = randn(Xoshiro(1234), Float32, 10)
dcontainer = DistributedUtils.DistributedDataContainer(backend, data)

expected_len_10 = Int(ceil(length(data) / tworkers))
@test length(dcontainer) == expected_len_10

pad_count_10 = expected_len_10 * tworkers - length(data)
expected_sum_10 = sum(data) + (pad_count_10 > 0 ? sum(data[1:pad_count_10]) : 0.0f0)

dsum_10 = sum(Base.Fix1(MLUtils.getobs, dcontainer), 1:MLUtils.numobs(dcontainer))
@test DistributedUtils.allreduce!(backend, [dsum_10], +)[1] ≈ expected_sum_10
