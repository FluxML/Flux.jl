# ==============================================
#          inspired by avik-pal's Lux.jl
# ==============================================

module DistributedUtils

using ChainRulesCore: ChainRulesCore
using ..Flux: AbstractFluxDistributedBackend, MPIBackend, NCCLBackend
using MLDataDevices: get_device, AbstractDevice
using Functors: fmap
using MLUtils: MLUtils, numobs
using Optimisers: Optimisers, AbstractRule, Leaf
using Random: Random
using Setfield: @set!

const CRC = ChainRulesCore

const NCCL_Initialized = Ref(false)
const MPI_Initialized = Ref(false)

"""
    initialized(backend::Type{<:AbstractFluxDistributedBackend})

Check if the given backend is initialized.
"""
initialized(::Type{<:MPIBackend}) = MPI_Initialized[]
initialized(::Type{<:NCCLBackend}) = NCCL_Initialized[]

"""
    initialize(backend::Type{<:AbstractFluxDistributedBackend}; kwargs...)

Initialize the given backend. Users can supply `cuda_devices` and `amdgpu_devices` to
initialize the backend with the given devices. These can be set to `missing` to prevent
initialization of the given device type. If set to `nothing`, and the backend is functional
we assign GPUs in a round-robin fashion. Finally, a list of integers can be supplied to
initialize the backend with the given devices. A guardrail runs before `MPI.Init()` —
including the `MPI.Init()` performed while bootstrapping `NCCLBackend`: it errors when
the job was started by a PMIx/OpenMPI launcher (i.e. `PMIX_RANK` or
`OMPI_COMM_WORLD_RANK` is set in the environment) while MPICH is the loaded MPI
library, a known unsafe combination that aborts `MPI.Init()` at the native level.
System OpenMPI launched with PMIx passes the check. Pass `force=true` to the same
`initialize(...)` call (for either `MPIBackend` or `NCCLBackend`) to bypass the
guardrail entirely (expert use only).

Possible values for `backend` are:

  - `MPIBackend`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `NCCLBackend`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA
    communications.
"""
function initialize(backend::Type{<:AbstractFluxDistributedBackend}; kwargs...)
    # initialized(backend) && return
    __initialize(backend; kwargs...)
    return
end

function __initialize end

"""
    __check_launcher_compat(env; library, force=false, mpi_initialized=false)

Internal helper. Users should call `initialize` instead.

Pure, MPI-free compatibility check between the environment a process was launched
in and the name of the loaded MPI library (as reported by `MPI.MPI_LIBRARY`, e.g.
`"MPICH"`, `"OpenMPI"`, `"MPIwrapper"`). Used by the MPI backend guardrail,
which runs only before `MPI.Init()`.

Returns `nothing` when the launch environment is compatible with the loaded
library, or a non-empty `String` error message when a known unsafe mismatch is
detected. The only combination judged unsafe is MPICH loaded while a PMIx/OpenMPI
launcher started the job (`PMIX_RANK` and/or `OMPI_COMM_WORLD_RANK` set). System
OpenMPI with PMIx passes, non-MPICH non-OpenMPI libraries are never judged, and
PMI2/SLURM/PMI1-era variables alone never trigger an error.

Passing `force=true` (expert bypass) or `mpi_initialized=true` (the guard only
runs before `MPI.Init()`) always returns `nothing`.
"""
function __check_launcher_compat(env::AbstractDict;
        library::AbstractString, force::Bool=false, mpi_initialized::Bool=false)
    (force || mpi_initialized) && return nothing
    library == "MPICH" || return nothing
    trigger = [v for v in ("PMIX_RANK", "OMPI_COMM_WORLD_RANK") if haskey(env, v)]
    isempty(trigger) && return nothing

    vars = join(trigger, " and ")
    return string(
        "Distributed initialization failed: found ", vars, " in the environment, ",
        "but the loaded MPI library is ", library, ". PMIx/OpenMPI launchers set ",
        "these variables, and the loaded ", library,
        " build was not started by such a launcher, so `MPI.Init()` will likely ",
        "abort or hang at the native level. Relaunch with a launcher matching the ",
        "loaded MPI library (e.g. `mpiexecjl` for MPI.jl's default MPICH_jll, or ",
        "`srun --mpi=pmi2` on SLURM), configure MPI.jl to use your system MPI via ",
        "`MPIPreferences.use_system_binary()`, or bypass this check by passing ",
        "`force=true` to the same `initialize` call.")
end

"""
    get_distributed_backend(backend::Type{<:AbstractFluxDistributedBackend})

Get the distributed backend for the given backend type. Possible values are:

  - `MPIBackend`: MPI backend for distributed training. Requires `MPI.jl` to be installed.
  - `NCCLBackend`: NCCL backend for CUDA distributed training. Requires `CUDA.jl`,
    `MPI.jl`, and `NCCL.jl` to be installed. This also wraps `MPI` backend for non-CUDA
    communications.

!!! danger

    `initialize(backend; kwargs...)` must be called before calling this function.
"""
function get_distributed_backend(backend::Type{<:AbstractFluxDistributedBackend})
    initialized(backend) ||
        error("Backend `$(backend)` is not initialized. Call `DistributedUtils.initialize` first.")
    return __get_distributed_backend(backend)
end

function __get_distributed_backend end

CRC.@non_differentiable get_distributed_backend(::Any...)

"""
    local_rank(backend::AbstractFluxDistributedBackend)

Get the local rank for the given backend.
"""
function local_rank end

CRC.@non_differentiable local_rank(::Any...)

"""
    total_workers(backend::AbstractFluxDistributedBackend)

Get the total number of workers for the given backend.
"""
function total_workers end

CRC.@non_differentiable total_workers(::Any...)

"""
    bcast!(backend::AbstractFluxDistributedBackend, sendrecvbuf; root::Int=0)
    bcast!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf; root::Int=0)

Backend Agnostic API to broadcast the given buffer `sendrecvbuf` or `sendbuf` to all
workers into `recvbuf`. The value at `root` will be broadcasted to all other workers.
"""
function bcast!(backend::AbstractFluxDistributedBackend, sendrecvbuf; root::Int=0)
    return __bcast!(backend, sendrecvbuf, get_device(sendrecvbuf); root)
end

function bcast!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf; root::Int=0)
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev == recv_dev
        return __bcast!(backend, sendbuf, recvbuf, send_dev; root)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        return __bcast!(backend, sendbuf_, recvbuf, recv_dev; root)
    end
end

function __bcast! end

CRC.@non_differentiable bcast!(::Any...)

function avg end

"""
    allreduce!(backend::AbstractFluxDistributedBackend, sendrecvbuf, op)
    allreduce!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf, op)

Backend Agnostic API to perform an allreduce operation on the given buffer `sendrecvbuf` or
`sendbuf` and store the result in `recvbuf`.

`op` allows a special `DistributedUtils.avg` operation that averages the result across all
workers.
"""
function allreduce!(backend::AbstractFluxDistributedBackend, sendrecvbuf, op::F) where {F}
    return __allreduce!(backend, sendrecvbuf, op, get_device(sendrecvbuf))
end

function allreduce!(
        backend::AbstractFluxDistributedBackend, sendbuf, recvbuf, op::F) where {F}
        send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev == recv_dev
        __allreduce!(backend, sendbuf, recvbuf, op, send_dev)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        __allreduce!(backend, sendbuf_, recvbuf, op, recv_dev)
    end
    return
end

function __allreduce! end

CRC.@non_differentiable allreduce!(::Any...)

"""
    reduce!(backend::AbstractFluxDistributedBackend, sendrecvbuf, op; root::Int=0)
    reduce!(backend::AbstractFluxDistributedBackend, sendbuf, recvbuf, op; root::Int=0)

Backend Agnostic API to perform a reduce operation on the given buffer `sendrecvbuf` or
`sendbuf` and store the result in `recvbuf`.

`op` allows a special `DistributedUtils.avg` operation that averages the result across all
workers.
"""
function reduce!(
        backend::AbstractFluxDistributedBackend, sendrecvbuf, op::F; root::Int=0) where {F}
    return __reduce!(backend, sendrecvbuf, op, get_device(sendrecvbuf); root)
end

function reduce!(backend::AbstractFluxDistributedBackend,
        sendbuf, recvbuf, op::F; root::Int=0) where {F}
    send_dev = get_device(sendbuf)
    recv_dev = get_device(recvbuf)
    if send_dev == recv_dev
        __reduce!(backend, sendbuf, recvbuf, op, send_dev; root)
    else
        sendbuf_ = sendbuf |> recv_dev
        @warn "`sendbuf` and `recvbuf` are on different devices." maxlog=1
        __reduce!(backend, sendbuf_, recvbuf, op, recv_dev; root)
    end
    return
end

function __reduce! end

CRC.@non_differentiable reduce!(::Any...)

## As Flux model is an arbitrary type it's not possible to dispatch `synchronize!!`
## end user needs to wrap Flux model into `FluxDistributedModel`
## e.g. model = DistributedUtils.synchronize!!(backend, FluxDistributedModel(model); root=0)
struct FluxDistributedModel{M}
    model::M
end

# synchronize!
"""
    synchronize!!(backend::AbstractFluxDistributedBackend, ps; root::Int=0)

Synchronize the given structure `ps` using the given backend. The value at `root` will be
broadcasted to all other workers.
"""
function synchronize!!(backend::AbstractFluxDistributedBackend, model::FluxDistributedModel; root::Int=0)
    return fmap(x -> synchronize!!(backend, x; root), model.model)
end

function synchronize!!(backend::AbstractFluxDistributedBackend, ps::Tuple; root::Int=0)
    length(ps) == 0 && return ps
    return map(x -> synchronize!!(backend, x; root), ps)
end

function synchronize!!(backend::AbstractFluxDistributedBackend,
        ps::NamedTuple{fields}; root::Int=0) where {fields}
    length(ps) == 0 && return ps
    return NamedTuple{fields}(map(x -> synchronize!!(backend, x; root), values(ps)))
end

function synchronize!!(
        backend::AbstractFluxDistributedBackend, ps::AbstractArray{T}; root::Int=0) where {T}
    if isbitstype(T)
        bcast!(backend, ps; root)
        return ps
    end
    return map(x -> synchronize!!(backend, x; root), ps)
end

# if no method for a given type, just return the value
function synchronize!!(backend::AbstractFluxDistributedBackend, ps::T; root::Int=0) where {T}
    isbitstype(T) && return bcast!(backend, [ps]; root)[]
    return ps
end

# data container
"""
    DistributedDataContainer(backend::AbstractFluxDistributedBackend, data)

`data` must be compatible with `MLUtils` interface. The returned container is compatible
with `MLUtils` interface and is used to partition the dataset across the available
processes.

The container does not modify the underlying dataset. It only pads its internal index
sequence so that every process receives a shard of the same length. When the number of
observations `N = numobs(data)` is not divisible by the number of processes, the padding
is drawn by repeating indices cyclically from the start of the dataset (`mod1`-style),
keeping every index in `1:N`. Each shard therefore has length `cld(N, workers)`.

Because the padding duplicates observations, those duplicated observations are included
in training batches and therefore receive extra weight in the averaged gradients, which
changes the effective training objective. They also affect epoch accounting and any
metric computed over the sharded or padded data (for example epoch counts, losses, or
sums). Metrics meant to describe the original dataset must exclude or otherwise account
for the duplicates.

An empty dataset (`numobs(data) == 0`) is rejected with an `ArgumentError`.

!!! danger

    `MLUtils.jl` must be installed and loaded before using this.
"""
struct DistributedDataContainer
    data
    idxs
end

function DistributedDataContainer(backend::AbstractFluxDistributedBackend, data)
    return __construct_distributed_data_container(backend, data)
end

Base.length(ddc::DistributedDataContainer) = length(ddc.idxs)

Base.getindex(ddc::DistributedDataContainer, i) = getindex(ddc.data, ddc.idxs[i])

function MLUtils.getobs(dc::DistributedDataContainer, idx)
    return MLUtils.getobs(dc.data, dc.idxs[idx])
end

function __construct_distributed_data_container(
        backend::AbstractFluxDistributedBackend, data)
    total_size = numobs(data)
    split_across = total_workers(backend)

    total_size == 0 &&
        throw(ArgumentError("cannot build a DistributedDataContainer for an empty " *
                            "dataset (numobs(data) == 0)"))
    size_per_worker = cld(total_size, split_across)

    # Pad the index list so that it is evenly divisible by the number of workers.
    # Repeats are drawn cyclically from the start of the dataset, keeping every
    # index in `1:total_size` and the original indices `1:total_size` before any
    # repeats.
    total_padded = size_per_worker * split_across
    indices = [mod1(i, total_size) for i in 1:total_padded]

    partitions = collect(Iterators.partition(indices, size_per_worker))
    idxs = collect(partitions[local_rank(backend) + 1])

    return DistributedDataContainer(data, idxs)
end

# Distributed Optimizer
"""
    DistributedOptimizer(backend::AbstractFluxDistributedBackend, optimizer)

Wrap the `optimizer` in a `DistributedOptimizer`. Before updating the parameters, this
averages the gradients across the processes using Allreduce.

## Arguments

  - `optimizer`: An Optimizer compatible with the Optimisers.jl package

"""
struct DistributedOptimizer{B <: AbstractFluxDistributedBackend} <: AbstractRule
    backend::B
    opt
end

function Optimisers.apply!(opt::DistributedOptimizer, state, x, y)
    y_avg = DistributedUtils.allreduce!(opt.backend, y, DistributedUtils.avg)
    return Optimisers.apply!(opt.opt, state, x, y_avg)
end

Optimisers.init(opt::DistributedOptimizer, x::AbstractArray) = Optimisers.init(opt.opt, x)

function Optimisers._adjust(opt::DistributedOptimizer, nt::NamedTuple)
    return DistributedOptimizer(opt.backend, Optimisers._adjust(opt.opt, nt))
end

function DistributedUtils.synchronize!!(
        backend::AbstractFluxDistributedBackend, ps::Leaf; root::Int=0)
    @set! ps.state = DistributedUtils.synchronize!!(backend, ps.state; root)
    return ps
end

end
