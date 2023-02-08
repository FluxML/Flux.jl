import Adapt: adapt, adapt_storage
using  LinearAlgebra: Cholesky
using Zygote: IdSet
import Functors: Functors, @functor, functor, fmap, isleaf
using SparseArrays: AbstractSparseArray

"""
    testmode!(m, mode = true)

Set a layer or model's test mode (see below).
Using `:auto` mode will treat any gradient computation as training.

_Note_: if you manually set a model into test mode, you need to manually place
it back into train mode during training phase.

Possible values include:
- `false` for training
- `true` for testing
- `:auto` or `nothing` for Flux to detect the mode automatically
"""
testmode!(m, mode = true) = (foreach(x -> testmode!(x, mode), trainable(m)); m)

"""
    trainmode!(m, mode = true)

Set a layer of model's train mode (see below).
Symmetric to [`testmode!`](@ref) (i.e. `trainmode!(m, mode) == testmode!(m, !mode)`).

_Note_: if you manually set a model into train mode, you need to manually place
it into test mode during testing phase.

Possible values include:
- `true` for training
- `false` for testing
- `:auto` or `nothing` for Flux to detect the mode automatically
"""
trainmode!(m, mode = true) = mode isa Bool ? testmode!(m, !mode) : testmode!(m, mode)

function params!(p::Params, x, seen = IdSet())
  if x isa AbstractArray{<:Number} && Functors.isleaf(x)
    return push!(p, x)
  elseif x in seen
    nothing
  else
    push!(seen, x)
    for child in trainable(x)
      params!(p, child, seen)
    end
  end
end

"""
    params(model)
    params(layers...)

Given a model or specific layers from a model, create a `Params` object pointing to its trainable parameters.

This can be used with the `gradient` function, see [Taking Gradients](@ref), or as input to the [`Flux.train!`](@ref Flux.train!) function.

The behaviour of `params` on custom types can be customized using [`Functors.@functor`](@ref) or [`Flux.trainable`](@ref).

# Examples
```jldoctest
julia> using Flux: params

julia> params(Chain(Dense(ones(2,3)), softmax))  # unpacks Flux models
Params([[1.0 1.0 1.0; 1.0 1.0 1.0], [0.0, 0.0]])

julia> bn = BatchNorm(2, relu)
BatchNorm(2, relu)  # 4 parameters, plus 4 non-trainable

julia> params(bn)  # only the trainable parameters
Params([Float32[0.0, 0.0], Float32[1.0, 1.0]])

julia> params([1, 2, 3], [4])  # one or more arrays of numbers
Params([[1, 2, 3], [4]])

julia> params([[1, 2, 3], [4]])  # unpacks array of arrays
Params([[1, 2, 3], [4]])

julia> params(1, [2 2], (alpha=[3,3,3], beta=Ref(4), gamma=sin))  # ignores scalars, unpacks NamedTuples
Params([[2 2], [3, 3, 3]])
```
"""
function params(m...)
  ps = Params()
  params!(ps, m)
  return ps
end

# Allows caching of the parameters when params is called within gradient() to fix #2040.
# @non_differentiable params(m...)  # https://github.com/FluxML/Flux.jl/pull/2054
# That speeds up implicit use, and silently breaks explicit use. 
# From @macroexpand Zygote.@nograd params(m...) and https://github.com/FluxML/Zygote.jl/pull/1248
Zygote._pullback(::Zygote.Context{true}, ::typeof(params), m...) = params(m), _ -> nothing

struct FluxCUDAAdaptor end
adapt_storage(to::FluxCUDAAdaptor, x) = CUDA.cu(x)
adapt_storage(to::FluxCUDAAdaptor, x::Zygote.FillArrays.AbstractFill) = CUDA.cu(collect(x))
if VERSION >= v"1.7"
  adapt_storage(to::FluxCUDAAdaptor, x::Random.TaskLocalRNG) = CUDA.default_rng()
else
  adapt_storage(to::FluxCUDAAdaptor, x::Random._GLOBAL_RNG) = CUDA.default_rng()
end
adapt_storage(to::FluxCUDAAdaptor, x::CUDA.RNG) = x
adapt_storage(to::FluxCUDAAdaptor, x::AbstractRNG) =
  error("Cannot map RNG of type $(typeof(x)) to GPU. GPU execution only supports Random.default_rng().")

# TODO: figure out the correct design for OneElement
adapt_storage(to::FluxCUDAAdaptor, x::Zygote.OneElement) = CUDA.cu(collect(x))

struct FluxCPUAdaptor end

# define rules for handling structured arrays
adapt_storage(to::FluxCPUAdaptor, x::AbstractArray) = adapt(Array, x)
adapt_storage(to::FluxCPUAdaptor, x::AbstractRange) = x
adapt_storage(to::FluxCPUAdaptor, x::Zygote.FillArrays.AbstractFill) = x
adapt_storage(to::FluxCPUAdaptor, x::T) where T <: CUDA.CUSPARSE.CUDA.CUSPARSE.AbstractCuSparseMatrix = adapt(Array, x)
adapt_storage(to::FluxCPUAdaptor, x::Zygote.OneElement) = x
adapt_storage(to::FluxCPUAdaptor, x::AbstractSparseArray) = x
adapt_storage(to::FluxCPUAdaptor, x::CUDA.RNG) = Random.default_rng()
adapt_storage(to::FluxCPUAdaptor, x::AbstractRNG) = x

function ChainRulesCore.rrule(::typeof(Adapt.adapt_storage), to::FluxCPUAdaptor, x::CUDA.AbstractGPUArray)
  adapt_storage(to, x), dx -> (NoTangent(), NoTangent(), adapt_storage(FluxCUDAAdaptor(), unthunk(dx)))
end

# The following rrules for adapt are here to avoid double wrapping issues
# as seen in https://github.com/FluxML/Flux.jl/pull/2117#discussion_r1027321801

ChainRulesCore.rrule(::typeof(adapt), a::FluxCPUAdaptor, x::AnyCuArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), adapt(FluxCUDAAdaptor(), unthunk(Δ)))

ChainRulesCore.rrule(::typeof(adapt), a::FluxCPUAdaptor, x::AbstractArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), Δ)

ChainRulesCore.rrule(::typeof(adapt), a::FluxCUDAAdaptor, x::AnyCuArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), Δ)

ChainRulesCore.rrule(::typeof(adapt), a::FluxCUDAAdaptor, x::AbstractArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), adapt(FluxCPUAdaptor(), unthunk(Δ)))


# CPU/GPU movement conveniences

"""
    cpu(m)

Copies `m` onto the CPU, the opposite of [`gpu`](@ref).
Recurses into structs marked [`@functor`](@ref).

# Example
```julia-repl
julia> m_gpu = Dense(CUDA.randn(2, 5))
Dense(5 => 2)       # 12 parameters

julia> m_gpu.bias  # matches the given weight matrix
2-element CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}:
 0.0
 0.0

julia> m = m_gpu |> cpu
Dense(5 => 2)       # 12 parameters

julia> m.bias
2-element Vector{Float32}:
 0.0
 0.0
```
"""
cpu(x) = fmap(x -> adapt(FluxCPUAdaptor(), x), x, exclude = _isleaf)

_isbitsarray(::AbstractArray{<:Number}) = true
_isbitsarray(::AbstractArray{T}) where T = isbitstype(T)
_isbitsarray(x) = false

_isleaf(::AbstractRNG) = true
_isleaf(x) = _isbitsarray(x) || Functors.isleaf(x)

"""
    gpu(x)

Copies `m` to the current GPU device, if one is available.
If no GPU is available, it does nothing (but prints a warning the first time).

On arrays, this calls CUDA's `cu`, which also changes arrays
with Float64 elements to Float32 while copying them to the device.
To act on arrays within a struct, the struct type must be marked with [`@functor`](@ref).

Use [`cpu`](@ref) to copy back to ordinary `Array`s.
See also [`f32`](@ref) and [`f16`](@ref) to change element type only.

See the [CUDA.jl docs](https://juliagpu.github.io/CUDA.jl/stable/usage/multigpu/) 
to help identify the current device.

# Example
```julia-repl
julia> m = Dense(rand(2, 3))  # constructed with Float64 weight matrix
Dense(3 => 2)       # 8 parameters

julia> typeof(m.weight)
Matrix{Float64} (alias for Array{Float64, 2})

julia> m_gpu = gpu(m)  # can equivalently be written m_gpu = m |> gpu
Dense(3 => 2)       # 8 parameters

julia> typeof(m_gpu.weight)
CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
```
"""
function gpu(x)
  check_use_cuda()
  use_cuda[] ? fmap(x -> Adapt.adapt(FluxCUDAAdaptor(), x), x; exclude = _isleaf) : x
end

function check_use_cuda()
  if use_cuda[] === nothing
    use_cuda[] = CUDA.functional()
    if !(use_cuda[])
      @info """The GPU function is being called but the GPU is not accessible. 
               Defaulting back to the CPU. (No action is required if you want to run on the CPU).""" maxlog=1
    end
  end
end
ChainRulesCore.@non_differentiable check_use_cuda()

# Precision

adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Real}) = convert.(T, xs) # piracy

paramtype(T::Type{<:Real}, m) = fmap(x -> adapt(T, x), m)

"""
    f32(m)

Converts the `eltype` of model's parameters to `Float32` (which is Flux's default).
Recurses into structs marked with [`@functor`](@ref).
See also [`f64`](@ref) and [`f16`](@ref).
"""
f32(m) = paramtype(Float32, m)

"""
    f64(m)

Converts the `eltype` of model's parameters to `Float64`.
Recurses into structs marked with [`@functor`](@ref).
"""
f64(m) = paramtype(Float64, m)

"""
    f16(m)

Converts the `eltype` of model's parameters to `Float16`.
Recurses into structs marked with [`@functor`](@ref).

Support for `Float16` is limited on many CPUs. Julia may
convert to `Float32` for each operation, which is slow.

# Example
```jldoctest
julia> m = Chain(Dense(784, 2048, relu), Dense(2048, 10))  # all Float32
Chain(
  Dense(784 => 2048, relu),             # 1_607_680 parameters
  Dense(2048 => 10),                    # 20_490 parameters
)                   # Total: 4 arrays, 1_628_170 parameters, 6.211 MiB.

julia> m |> f16  # takes half the memory
Chain(
  Dense(784 => 2048, relu),             # 1_607_680 parameters
  Dense(2048 => 10),                    # 20_490 parameters
)                   # Total: 4 arrays, 1_628_170 parameters, 3.106 MiB.
```
"""
f16(m) = paramtype(Float16, m)

# Functors for certain Julia data structures
@functor Cholesky
trainable(c::Cholesky) = ()

