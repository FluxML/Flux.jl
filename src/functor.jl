import Adapt: adapt, adapt_storage
using  LinearAlgebra: Cholesky
using Zygote: IdSet
import Functors: Functors, @functor, functor, fmap, isleaf
using SparseArrays: AbstractSparseArray

trainable(m) = functor(m)[1]

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

params!(p::Params, x::AbstractArray{<:Number}, seen = IdSet()) = push!(p, x)

function params!(p::Params, x, seen = IdSet())
  x in seen && return
  push!(seen, x)
  for child in trainable(x)
    params!(p, child, seen)
  end
end

"""
    params(model)

Given a model or specific layers from a model, create a `Params` object pointing to its trainable parameters.

This can be used with [`gradient`](@ref), or as input to the [`Flux.train!`](@ref Flux.train!) function.

# Examples
```jldoctest
julia> params(Chain(Dense(ones(2,3))), softmax)
Params([[1.0 1.0 1.0; 1.0 1.0 1.0], [0.0, 0.0]])

julia> params(BatchNorm(2, relu))
Params([Float32[0.0, 0.0], Float32[1.0, 1.0]])
```
"""
function params(m...)
  ps = Params()
  params!(ps, m)
  return ps
end

function loadparams!(m, xs)
  for (p, x) in zip(params(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end

struct FluxCUDAAdaptor end
adapt_storage(to::FluxCUDAAdaptor, x) = CUDA.cu(x)
adapt_storage(to::FluxCUDAAdaptor, x::Zygote.FillArrays.AbstractFill) = CUDA.cu(collect(x))

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

Zygote.@adjoint function Array(x::CUDA.CuArray)
  Array(x), d -> (CUDA.cu(d),)
end

Zygote.@adjoint function Adapt.adapt_storage(to::FluxCPUAdaptor, x::CUDA.AbstractGPUArray)
  adapt_storage(to, x), d -> (nothing, adapt_storage(FluxCUDAAdaptor(), d),)
end

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

"""
    gpu(x)

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
  check_use_cuda()
  use_cuda[] ? fmap(x -> Adapt.adapt(FluxCUDAAdaptor(), x), x; exclude = _isbitsarray) : x
end

function check_use_cuda()
  if use_cuda[] === nothing
    use_cuda[] = CUDA.functional()
    if use_cuda[] && !CUDA.has_cudnn()
      @warn "CUDA.jl found cuda, but did not find libcudnn. Some functionality will not be available."
    end
  end
end
Zygote.@nograd check_use_cuda

# Precision

adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Real}) = convert.(T, xs) # piracy

paramtype(T::Type{<:Real}, m) = fmap(x -> adapt(T, x), m)

"""
    f32(m)

Convert the `eltype` of model's parameters to `Float32`.
"""
f32(m) = paramtype(Float32, m)

"""
    f64(m)

Convert the `eltype` of model's parameters to `Float64`.
"""
f64(m) = paramtype(Float64, m)

# Functors for certain Julia data structures
@functor Cholesky
trainable(c::Cholesky) = ()
