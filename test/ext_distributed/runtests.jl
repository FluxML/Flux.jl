# Distributed Tests
using MPI, Pkg, Test
if get(ENV, "FLUX_TEST_DISTRIBUTED_NCCL", "false") == "true"
    import CUDA
end

nprocs_str = get(ENV, "JULIA_MPI_TEST_NPROCS", "")
nprocs = nprocs_str == "" ? clamp(Sys.CPU_THREADS, 2, 4) : parse(Int, nprocs_str)
testdir = @__DIR__
isdistributedtest(f) = endswith(f, "_distributedtest.jl")
distributedtestfiles = String[]
for (root, dirs, files) in walkdir(testdir)
    for file in files
        if isdistributedtest(file)
            push!(distributedtestfiles, joinpath(root, file))
        end
    end
end

@info "Running Distributed Tests with $nprocs processes"

cur_proj = dirname(Pkg.project().path)

@testset "Distributed" begin
    backends = get(ENV, "FLUX_TEST_DISTRIBUTED_NCCL", "false") == "true" ? ("mpi", "nccl") : ("mpi",)
    for backend_type in backends
        np = backend_type == "nccl" ? min(nprocs, length(CUDA.devices())) : nprocs
        @testset "Backend: $(backend_type)" begin
            @testset "$(basename(file))" for file in distributedtestfiles
                @info "Running $file with $backend_type backend"
                run(`$(MPI.mpiexec()) -n $(np) $(Base.julia_cmd()) --color=yes \
                     --code-coverage=user --project=$(cur_proj) --startup-file=no $(file) \
                     $(backend_type)`)
                Test.@test true
            end
        end
    end
end