import Adapt: adapt, adapt_storage
using  LinearAlgebra: Cholesky
using Zygote: IdSet
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


# Flattening models to weight vectors, and back

function _restructure(m, xs)
  i = 0
  filter = (x, c) -> any(y -> c === y, trainable(x))
  walk = filtered_walk(filter)
  m̄ = fmap(m; walk) do x
    x isa AbstractArray{<:Number} || return x
    x = reshape(xs[i .+ (1:length(x))], size(x))
    i += length(x)
    return x
  end
  length(xs) == i || @warn "Expected $(i) params, got $(length(xs))"
  return m̄
end

@adjoint function _restructure(m, xs)
  m̄, numel = _restructure(m, xs), length(xs)
  function _restructure_pullback(dm)
    xs′ = destructure(dm)[1]
    numel == length(xs′) || @warn "Expected $(numel) params, got $(length(xs′))"
    return (nothing, xs′)
  end
  return m̄, _restructure_pullback
end

"""
    destructure(m)
Flatten a model's parameters into a single weight vector.
    julia> m = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
    Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
    julia> θ, re = destructure(m);
    julia> θ
    67-element Vector{Float32}:
    -0.1407104
    ...
The second return value `re` allows you to reconstruct the original network after making
modifications to the weight vector (for example, with a hypernetwork).
    julia> re(θ .* 2)
    Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
"""
function destructure(m)
  xs = Zygote.Buffer([])
  collect_params!(xs, m)
  return vcat(vec.(copy(xs))...), p -> _restructure(m, p)
end

function collect_params!(xs, m)
  filter = (x, c) -> any(y -> c === y, trainable(x))
  walk = filtered_walk(filter)
  fmap(m; walk) do x
    x isa AbstractArray{<:Number} && push!(xs, x)
    return x
  end
end

function filtered_walk(filter)
  seen = IdSet()

  function walk(f, x)
    x in seen && return x
    push!(seen, x)

    children, reconstruct = functor(x)
    mappedchildren = map(children) do c
      filter(x, c) ? f(c) : c
    end
    reconstruct(mappedchildren)
  end

  return walk
end


"""
  params(m...)

Collect trainable parameters (a.k.a. numerical arrays)
from the input model(s) `m` into a [`Zygote.Params`](@ref) object. 

Only the parameters that can be reached by recursion 
on the [`trainable`](@ref) children of
the tree with root `m` are collected.

# Usage

```julia-repl 
julia> m = Dense(ones(2, 3), zeros(2))
Dense(3, 2)         # 8 parameters

julia> ps = Flux.params(m)
Params([[1.0 1.0 1.0; 1.0 1.0 1.0], [0.0, 0.0]])

julia> x = ones(3)
3-element Vector{Float64}:
 1.0
 1.0
 1.0

julia> gs = gradient(() -> sum(2 .* m(x)), ps)
Grads(...)

julia> gs[m.weight]
2×3 Matrix{Float64}:
 2.0  2.0  2.0
 2.0  2.0  2.0
```
"""
function params end

## TODO This causes some test regressions. Why?
# function params(m...)
#   ps = Params()
#   collect_params!(ps, m)
#   return ps
# end

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
    params(layers...)

Given a model or specific layers from a model, create a `Params` object pointing to its trainable parameters.

This can be used with the `gradient` function, see [Taking Gradients](@ref), or as input to the [`Flux.train!`](@ref Flux.train!) function.

The behaviour of `params` on custom types can be customized using [`Functor.@functor`](@ref) or [`Flux.trainable`](@ref).

# Examples
```jldoctest
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
    if !(use_cuda[])
      @info """The GPU function is being called but the GPU is not accessible. 
               Defaulting back to the CPU. (No action is required if you want to run on the CPU).""" maxlog=1
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
