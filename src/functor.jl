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

struct FluxCPUAdaptor end

# define rules for handling structured arrays
adapt_storage(to::FluxCPUAdaptor, x::AbstractArray) = adapt(Array, x)
adapt_storage(to::FluxCPUAdaptor, x::AbstractRange) = x
adapt_storage(to::FluxCPUAdaptor, x::Zygote.FillArrays.AbstractFill) = x
adapt_storage(to::FluxCPUAdaptor, x::Zygote.OneElement) = x
adapt_storage(to::FluxCPUAdaptor, x::AbstractSparseArray) = x
adapt_storage(to::FluxCPUAdaptor, x::AbstractRNG) = x

# CPU/GPU movement conveniences

"""
    cpu(m)

Moves `m` onto the CPU, the opposite of [`gpu`](@ref).
Recurses into structs marked [`@functor`](@ref).

```julia-repl
julia> m = Dense(1,2)
Dense(1, 2)

julia> m_gpu = gpu(m)
Dense(1, 2)

julia> typeof(m_gpu.W)
CuArray{Float32, 2}

julia> m_cpu = cpu(m_gpu)
Dense(1, 2)

julia> typeof(m_cpu.W)
Matrix{Float32}
```
"""
cpu(x) = fmap(x -> adapt(FluxCPUAdaptor(), x), x)

_isbitsarray(::AbstractArray{<:Number}) = true
_isbitsarray(::AbstractArray{T}) where T = isbitstype(T)
_isbitsarray(x) = false

_isleaf(::AbstractRNG) = true
_isleaf(x) = _isbitsarray(x) || Functors.isleaf(x)

"""
    gpu(x)

Requires CUDA and NNlibCUDA to be loaded
```julia-rept
julia> using Flux, CUDA, NNlibCUDA
```
Moves `m` to the current GPU device, if available. It is a no-op otherwise.
See the [CUDA.jl docs](https://juliagpu.github.io/CUDA.jl/stable/usage/multigpu/)
to help identify the current device.

This works for functions, and any struct marked with [`@functor`](@ref).

```julia-repl
julia> m = Dense(1,2)
Dense(1, 2)

julia> typeof(m.W)
Matrix{Float32}

julia> m_gpu = gpu(m)
Dense(1, 2)

julia> typeof(m_gpu.W) # notice the type of the array changed to a CuArray
CuArray{Float32, 2}
```
"""
function gpu(x)
  try
    _gpu(x)
  catch
    @info """
    The GPU functionality is being called via `Flux.gpu` but `NNlibCUDA`
    must be loaded to access GPU functionality""" maxlog=1
    return x
  end
end

function _gpu end

# Precision

adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Real}) = convert.(T, xs) # piracy

paramtype(T::Type{<:Real}, m) = fmap(x -> adapt(T, x), m)

"""
    f32(m)

Converts the `eltype` of model's parameters to `Float32` (which is Flux's default).
Recurses into structs marked with [`@functor`](@ref).
"""
f32(m) = paramtype(Float32, m)

"""
    f64(m)

Converts the `eltype` of model's parameters to `Float64`.
Recurses into structs marked with [`@functor`](@ref).
"""
f64(m) = paramtype(Float64, m)

# Functors for certain Julia data structures
@functor Cholesky
trainable(c::Cholesky) = ()

