# ==============================================
#          inspired by avik-pal's Lux.jl
# ==============================================

abstract type AbstractFluxDistributedBackend end

"""
    MPIBackend(comm = nothing)

Create an MPI backend for distributed training. Users should not use this function directly.
Instead use [`DistributedUtils.get_distributed_backend(MPIBackend)`](@ref).
"""
struct MPIBackend{C} <: AbstractFluxDistributedBackend
    comm::C

    function MPIBackend(comm=nothing)
        if Base.get_extension(@__MODULE__, :FluxMPIExt) === nothing
            error("`MPIBackend` requires `MPI.jl` to be loaded.")
        end
        return new{typeof(comm)}(comm)
    end
end

"""
    NCCLBackend(comm = nothing, mpi_backend = nothing)

Create an NCCL backend for distributed training. Users should not use this function
directly. Instead use [`DistributedUtils.get_distributed_backend(NCCLBackend)`](@ref).
"""
struct NCCLBackend{C, M <: Union{Nothing, MPIBackend}} <: AbstractFluxDistributedBackend
    comm::C
    mpi_backend::M

    function NCCLBackend(comm=nothing, mpi_backend=nothing)
        if Base.get_extension(@__MODULE__, :FluxMPINCCLExt) === nothing
            error("`NCCLBackend` requires `CUDA.jl`, `MPI.jl` and `NCCL.jl` to be loaded.")
        end
        return new{typeof(comm), typeof(mpi_backend)}(comm, mpi_backend)
    end
end

# Preferences for GPU-Aware MPI
const MPI_CUDA_AWARE = @load_preference("FluxDistributedMPICUDAAware", false)
const MPI_ROCM_AWARE = @load_preference("FluxDistributedMPIROCMAware", false)