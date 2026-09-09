using Test
using Pkg
using MPI

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
sort!(distributedtestfiles) # Deterministic order

@info "Running Distributed Tests with $nprocs processes"

cur_proj = dirname(Pkg.project().path)
# Use a watchdog timeout (e.g., 2 minutes) for each distributed test file
timeout_seconds = parse(Float64, get(ENV, "FLUX_TEST_DISTRIBUTED_TIMEOUT", "120.0"))

@testset "Distributed" begin
    backends = get(ENV, "FLUX_TEST_DISTRIBUTED_NCCL", "false") == "true" ? ("mpi", "nccl") : ("mpi",)
    for backend_type in backends
        np = backend_type == "nccl" ? min(nprocs, length(CUDA.devices())) : nprocs
        @testset "Backend: $(backend_type)" begin
            @testset "$(basename(file))" for file in distributedtestfiles
                @info "Running $file with $backend_type backend " *
                      "(env FLUX_TEST_DISTRIBUTED_BACKEND=$backend_type)"
                
                cmd = `$(MPI.mpiexec()) -n $(np) $(Base.julia_cmd()) --color=yes --project=$(cur_proj) --startup-file=no $(file)`
                # Backend per child via env var FLUX_TEST_DISTRIBUTED_BACKEND
                # (not a positional arg): one parent runs both backend passes.
                cmd = addenv(cmd, "FLUX_TEST_DISTRIBUTED_BACKEND" => backend_type)
                
                # Explicitly inherit the child streams so the child's output
                # and errors are visible in the CI logs.
                proc = run(pipeline(cmd, stdout=stdout, stderr=stderr), wait=false)
                
                # Watchdog loop
                start_time = time()
                is_timeout = false
                while process_running(proc)
                    if time() - start_time > timeout_seconds
                        is_timeout = true
                        @error "Test $(basename(file)) timed out after $timeout_seconds seconds. Sending SIGINT for stack dump."
                        # Send SIGINT for stack dump before killing (Julia dumps backtrace on SIGINT)
                        run(ignorestatus(`kill -INT $(getpid(proc))`))
                        sleep(2) # Give it a moment to dump
                        kill(proc)
                        break
                    end
                    sleep(0.5)
                end
                
                if is_timeout
                    @test false # Fail the test due to timeout
                else
                    wait(proc)
                    # If the child process exit code is non-zero, fail the test. 
                    # Note: child processes MUST call exit(1) on failure.
                    @test proc.exitcode == 0
                end
            end
        end
    end
end
