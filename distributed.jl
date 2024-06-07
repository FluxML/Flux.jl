## RUN THIS WITH `mpiexecjl --project=@. -n 3 julia distributed.jl`

# using MPI
# MPI.Init()
# comm = MPI.COMM_WORLD
# println("Hello world, I am $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")
# MPI.Barrier(comm)


### FluxMPI.Init() ###########

using MPI, Flux, CUDA
using Dates
CUDA.allowscalar(false) 

FluxMPI_Initialized = Ref{Bool}(false)


"""
    Initialized()

Has FluxMPI been initialized?
"""
Initialized() = FluxMPI_Initialized[]


"""
    total_workers()

Get the total number of workers.
"""
@inline function total_workers()
  !Initialized() && throw(FluxMPINotInitializedError())
  return MPI.Comm_size(MPI.COMM_WORLD)
end

# CRC.@non_differentiable total_workers()


"""
    local_rank()

Get the rank of the process.
"""
@inline function local_rank()
  !Initialized() && throw(FluxMPINotInitializedError())
  return MPI.Comm_rank(MPI.COMM_WORLD)
end

# CRC.@non_differentiable local_rank()

"""
    Init(; gpu_devices::Union{Nothing,Vector{Int}}=nothing, verbose::Bool=false)

Setup `FluxMPI`. If GPUs are available and CUDA is functional, each rank is allocated a
GPU in a round-robin fashion.

If calling this function, no need to call `MPI.Init` first.
"""
function Init(; gpu_devices::Union{Nothing, Vector{Int}}=nothing, verbose::Bool=false)
    # if Initialized()
    #     verbose && fluxmpi_println("FluxMPI already initialized; Skipping...")
    #     return
    # end

    !MPI.Initialized() && MPI.Init()
    FluxMPI_Initialized[] = true

    if verbose && total_workers() == 1
        @warn "Using FluxMPI with only 1 worker. It might be faster to run the code without MPI" maxlog=1
    end
    println("HERE 0")
    rank = local_rank()

    println("HERE 1")
    if CUDA.functional()
        println("HERE 2")
        gpu_device = if gpu_devices === nothing
                        device_count = length(CUDA.devices())
                        (rank + 1) % device_count
                    else
                        gpu_devices[rank + 1]
                    end
        println("HERE 3")
        # verbose && fluxmpi_println("Using GPU $gpu_device")
        println("HERE 4")
        verbose && println("Using GPU $gpu_device")
        CUDA.device!(gpu_device)
        print("HERE 5")
    else
        verbose && println("Using CPU")
        # verbose && fluxmpi_println("Using CPU")
    end
end


# Print Functions
for print_fn in (:println, :print)
    function_name = Symbol("fluxmpi_" * string(print_fn))
    @eval begin
      function $(function_name)(args...; kwargs...)
        if !Initialized()
          $(print_fn)("$(Dates.now()) ", args...; kwargs...)
          return
        end
        rank = local_rank()
        size = total_workers()
        if size == 1
          $(print_fn)(args...; kwargs...)
          return
        end
        for r in 0:(size - 1)
          if r == rank
            $(print_fn)("$(Dates.now()) [$(rank) / $(size)] ", args...; kwargs...)
            flush(stdout)
          end
          MPI.Barrier(MPI.COMM_WORLD)
        end
        return
      end
  
    #   CRC.@non_differentiable $(function_name)(::Any...)
    end
  end
