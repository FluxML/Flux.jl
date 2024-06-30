using Flux, MPI, NCCL, CUDA
using Random
using Optimisers
using Zygote
using Statistics

CUDA.allowscalar(false)

# ==============================================
#               MPI backend
# ==============================================

CUDA.device!(0)

DistributedUtils.initialize(MPIBackend)
backend = DistributedUtils.get_distributed_backend(MPIBackend)
rank = DistributedUtils.local_rank(backend)

device = Flux.get_device()
model = Chain(Dense(1 => 256, tanh), Dense(256 => 1))
model = model |> device

model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0) 


x = rand(Float32, 1, 16) |> device
y = x .^ 3 |> device

opt = DistributedUtils.DistributedOptimizer(backend, Optimisers.Adam(0.001f0))
st_opt = Optimisers.setup(opt, model)

loss(m, x, y) = mean((m(x) .- y).^2)
g = gradient(m -> loss(m, x, y), model)[1]

st_opt = DistributedUtils.synchronize!!(backend, st_opt; root=0) 

data = Flux.DataLoader((x, y) |> gpu, batchsize=64, shuffle=true);

Optimisers.update(st_opt, model, g)

t1 = time()

for epoch in 1:100
  global model, st_opt
  l, back = Zygote.pullback(loss, model) # MethodError: no method matching loss(::Chain{Tuple{Dense{typeof(identity), CuArray{…}, CuArray{…}}, Dense{typeof(identity), CuArray{…}, CuArray{…}}}})
  gs = back(one(l))[1]
  st_opt, model = Optimisers.update(st_opt, model, gs)
end

println(time() - t1)

# ==============================================
#               NNCCL backend
# ==============================================

# DistributedUtils.initialize(NCCLBackend)

# backend = DistributedUtils.get_distributed_backend(NCCLBackend) 
# rank = DistributedUtils.local_rank(backend)

# model = Chain(Dense(1 => 256, tanh), Dense(256 => 1))

# rng = Random.default_rng()
# Random.seed!(rng, rank)

# model = DistributedUtils.synchronize!!(backend, DistributedUtils.FluxDistributedModel(model); root=0) 

# x = rand(1, 16) |> gpu
# y = x .^ 3

# opt = DistributedUtils.DistributedOptimizer(backend, Optimisers.Adam(0.001f0))
# st_opt = Optimisers.setup(opt, model)

# loss(model) = sum(abs2, model(x) .- y)

# st_opt = DistributedUtils.synchronize!!(backend, st_opt; root=0)


# ==============================================
#          Carlo's code
# ==============================================

## RUN THIS WITH `mpiexecjl --project=@. -n 3 julia distributed.jl`

# using MPI
# MPI.Init()
# comm = MPI.COMM_WORLD
# println("Hello world, I am $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")
# MPI.Barrier(comm)

# using MPI, Flux, CUDA
# using Dates
# CUDA.allowscalar(false) 

# FluxMPI_Initialized = Ref{Bool}(false)

# ### FluxMPI.Init() ###########

# """
#     Initialized()

# Has FluxMPI been initialized?
# """
# Initialized() = FluxMPI_Initialized[]


# """
#     total_workers()

# Get the total number of workers.
# """
# @inline function total_workers()
#   !Initialized() && throw(FluxMPINotInitializedError())
#   return MPI.Comm_size(MPI.COMM_WORLD)
# end

# # CRC.@non_differentiable total_workers()


# """
#     local_rank()

# Get the rank of the process.
# """
# @inline function local_rank()
#   !Initialized() && throw(FluxMPINotInitializedError())
#   return MPI.Comm_rank(MPI.COMM_WORLD)
# end

# # CRC.@non_differentiable local_rank()

# """
#     Init(; gpu_devices::Union{Nothing,Vector{Int}}=nothing, verbose::Bool=false)

# Setup `FluxMPI`. If GPUs are available and CUDA is functional, each rank is allocated a
# GPU in a round-robin fashion.

# If calling this function, no need to call `MPI.Init` first.
# """
# function Init(; gpu_devices::Union{Nothing, Vector{Int}}=nothing, verbose::Bool=false)
#     # if Initialized()
#     #     verbose && fluxmpi_println("FluxMPI already initialized; Skipping...")
#     #     return
#     # end

#     !MPI.Initialized() && MPI.Init()
#     FluxMPI_Initialized[] = true

#     if verbose && total_workers() == 1
#         @warn "Using FluxMPI with only 1 worker. It might be faster to run the code without MPI" maxlog=1
#     end
#     println("HERE 0")
#     rank = local_rank()

#     println("HERE 1")
#     if CUDA.functional()
#         println("HERE 2")
#         gpu_device = if gpu_devices === nothing
#                         device_count = length(CUDA.devices())
#                         (rank + 1) % device_count
#                     else
#                         gpu_devices[rank + 1]
#                     end
#         println("HERE 3")
#         # verbose && fluxmpi_println("Using GPU $gpu_device")
#         println("HERE 4")
#         verbose && println("Using GPU $gpu_device")
#         CUDA.device!(gpu_device)
#         print("HERE 5")
#     else
#         verbose && println("Using CPU")
#         # verbose && fluxmpi_println("Using CPU")
#     end
# end


# # Print Functions
# for print_fn in (:println, :print)
#     function_name = Symbol("fluxmpi_" * string(print_fn))
#     @eval begin
#       function $(function_name)(args...; kwargs...)
#         if !Initialized()
#           $(print_fn)("$(Dates.now()) ", args...; kwargs...)
#           return
#         end
#         rank = local_rank()
#         size = total_workers()
#         if size == 1
#           $(print_fn)(args...; kwargs...)
#           return
#         end
#         for r in 0:(size - 1)
#           if r == rank
#             $(print_fn)("$(Dates.now()) [$(rank) / $(size)] ", args...; kwargs...)
#             flush(stdout)
#           end
#           MPI.Barrier(MPI.COMM_WORLD)
#         end
#         return
#       end
  
#     #   CRC.@non_differentiable $(function_name)(::Any...)
#     end
#   end
