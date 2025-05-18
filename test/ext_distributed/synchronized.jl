const input_args = length(ARGS) == 2 ? ARGS : ("CPU", "mpi")
const backend_type = input_args[2] == "nccl" ? NCCLBackend : MPIBackend
const dev = input_args[1] == "CPU" ? Flux.cpu : Flux.gpu

function __get_array_based_on_rank(backend, dims; root)
    DistributedUtils.local_rank(backend) == root && return ones(dims...)
    return zeros(dims...)
end

root = 0

DistributedUtils.initialize(backend_type)
backend = DistributedUtils.get_distributed_backend(backend_type)

# Named Tuple
gs = (
    a=(b=__get_array_based_on_rank(backend, (2, 3); root),
        c=__get_array_based_on_rank(backend, (2, 3); root)),
    d=__get_array_based_on_rank(backend, (2, 3); root)) |> dev

gs_ = DistributedUtils.synchronize!!(backend, gs; root)

@test all(gs_.a.b .== 1)
@test all(gs_.a.c .== 1)
@test all(gs_.d .== 1)

## optimisers
opt = Optimisers.Adam(0.001f0)
st_opt = Optimisers.setup(opt, gs)

if DistributedUtils.local_rank(backend) == root
    st_opt.a.b.state[1] .= 1
    st_opt.a.b.state[2] .= 1
    st_opt.a.c.state[1] .= 1
    st_opt.a.c.state[2] .= 1
    st_opt.d.state[1] .= 1
    st_opt.d.state[2] .= 1
end

st_opt = DistributedUtils.synchronize!!(backend, st_opt; root)

@test all(st_opt.a.b.state[1] .== 1)
@test all(st_opt.a.b.state[2] .== 1)
@test all(st_opt.a.c.state[1] .== 1)
@test all(st_opt.a.c.state[2] .== 1)
@test all(st_opt.d.state[1] .== 1)
@test all(st_opt.d.state[2] .== 1)

# Has no state
opt = Optimisers.Descent(0.001f0)
st_opt = Optimisers.setup(opt, gs)

@test_nowarn DistributedUtils.synchronize!!(backend, st_opt; root)

## ComponentArrays
gs = (
    a=(b=__get_array_based_on_rank(backend, (2, 3); root),
        c=__get_array_based_on_rank(backend, (2, 3); root)),
    d=__get_array_based_on_rank(backend, (2, 3); root))
cgs_ = DistributedUtils.synchronize!!(backend, gs; root)

@test all(cgs_.a.b .== 1)
@test all(cgs_.a.c .== 1)
@test all(cgs_.d .== 1)

# Tuple
gs = (
    (__get_array_based_on_rank(backend, (2, 3); root),
        __get_array_based_on_rank(backend, (2, 3); root)),
    __get_array_based_on_rank(backend, (2, 3); root)) |> dev

gs = DistributedUtils.synchronize!!(backend, gs; root)

@test all(gs[1][1] .== 1)
@test all(gs[1][2] .== 1)
@test all(gs[2] .== 1)

# Miscelleneous
x = nothing
x = DistributedUtils.synchronize!!(backend, x; root)
@test x === nothing

x = ifelse(root == DistributedUtils.local_rank(backend), :x, :y)
x_ = DistributedUtils.synchronize!!(backend, x; root)
# Symbol should not change
@test x_ == x

x = DistributedUtils.synchronize!!(backend, DistributedUtils.local_rank(backend); root)
@test x == root