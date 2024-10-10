import Adapt: adapt, adapt_storage
using  LinearAlgebra: Cholesky
using Zygote: IdSet
import Functors: Functors, @functor, functor, fmap, isleaf
using SparseArrays: AbstractSparseArray

"""
    testmode!(model, [mode]) -> model

Set a layer, or all layers in a model, to test mode.
This disables the effect of [`Dropout`](@ref) and
some other regularisation layers.

If you manually set a model into test mode, you need to manually place
it back into train mode during training phase, using [`trainmode!`](@ref).

There is an optional second argument, which takes a symbol `:auto` to
reset all layers back to the default automatic mode.

# Example

```jldoctest
julia> d = Dropout(0.3)
Dropout(0.3)

julia> testmode!(d)   # dropout is now always disabled
Dropout(0.3, active=false)

julia> trainmode!(d)  # dropout is now always enabled
Dropout(0.3, active=true)

julia> testmode!(d, :auto)  # back to default
Dropout(0.3)
```
"""
testmode!(m) = testmode!(m, true)

"""
    trainmode!(model) -> model

Set a layer, or all layers in a model, to training mode.
Opposite to [`testmode!`](@ref), see further details there.
"""
trainmode!(m) = testmode!(m, false)
trainmode!(m, mode::Symbol) = testmode!(m, mode)
trainmode!(m, ::Nothing) = testmode!(m, nothing)  # why do we have so much API?

"""
    testmode!(model, inactive)

This two-argument method is largely internal. It recurses into the `model`,
and until a method like `testmode!(d::Dropout, inactive)` alters the activity of a layer.
Custom layers can support manual `testmode!` / `trainmode!` switching
by defining such a method.

Possible values of  `inactive` are:
- `true` for testing, i.e. `active=false`
- `false` for training, same as [`trainmode!`](@ref)`(m)`
- `:auto` or `nothing` for Flux to detect training automatically.

!!! compat
    This method may be removed in a future breaking change, to separate
    the user-facing `testmode!` from the internal recursion.
"""
function testmode!(m, mode)
  inactive = if mode isa Symbol
    mode === :auto || throw(ArgumentError(lazy"testmode! accepts only the symbol :auto, got :$mode"))
    nothing
  elseif mode isa Union{Bool,Nothing}
    mode
  else
    throw(ArgumentError(lazy"testmode! does not accept $(repr(mode)) as the 2nd argument"))
  end
  foreach(x -> testmode!(x, inactive), trainable(m))
  m
end

function params!(p::Params, x, seen = IdSet())
  if x isa AbstractArray{<:Number} && Functors.isleaf(x)
    return push!(p, x)
  elseif x in seen
    nothing
  else
    _check_new_macro(x)  # complains if you used @functor not @layer
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

This can be used with the `gradient` function, see the [training section of the manual](@ref man-training), or as input to the [`Flux.train!`](@ref Flux.train!) function.

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
# From @macroexpand Zygote.@non_differentiable params(m...) and https://github.com/FluxML/Zygote.jl/pull/1248
Zygote._pullback(::Zygote.Context{true}, ::typeof(params), m...) = params(m), _ -> nothing

struct FluxCPUAdaptor end

# define rules for handling structured arrays
adapt_storage(to::FluxCPUAdaptor, x::AbstractArray) = adapt(Array, x)
adapt_storage(to::FluxCPUAdaptor, x::AbstractRange) = x
adapt_storage(to::FluxCPUAdaptor, x::Zygote.FillArrays.AbstractFill) = x
adapt_storage(to::FluxCPUAdaptor, x::Zygote.OneElement) = x
adapt_storage(to::FluxCPUAdaptor, x::AbstractSparseArray) = x
adapt_storage(to::FluxCPUAdaptor, x::AbstractRNG) = x


# The following rrules for adapt are here to avoid double wrapping issues
# as seen in https://github.com/FluxML/Flux.jl/pull/2117#discussion_r1027321801
ChainRulesCore.rrule(::typeof(adapt), a::FluxCPUAdaptor, x::AbstractArray) =
  adapt(a, x), Δ -> (NoTangent(), NoTangent(), Δ)



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

_isleaf(x) = Functors.isleaf(x)

_isleaf(::AbstractArray{<:Number}) = true
_isleaf(::AbstractArray{T}) where T = isbitstype(T)
_isleaf(::Union{Transpose, Adjoint, PermutedDimsArray}) = false

_isleaf(::AbstractRNG) = true

# the order below is important
const GPU_BACKENDS = ("CUDA", "AMDGPU", "Metal", "CPU")
const GPU_BACKEND_ORDER = Dict(collect(zip(GPU_BACKENDS, 1:length(GPU_BACKENDS))))
const GPU_BACKEND = load_preference(MLDataDevices, "gpu_backend", "CUDA")


"""
    gpu(m)

Copies `m` to the current GPU device (using current GPU backend), if one is available.
If no GPU is available, it does nothing (but prints a warning the first time).

On arrays, this calls CUDA's `cu`, which also changes arrays
with Float64 elements to Float32 while copying them to the device (same for AMDGPU).
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
    @static if GPU_BACKEND == "CUDA"
        gpu(FluxCUDAAdaptor(), x)
    elseif GPU_BACKEND == "AMD"
        @warn "\"AMD\" backend is deprecated. Please use \"AMDGPU\" instead." maxlog=1
        gpu(FluxAMDGPUAdaptor(), x)
    elseif GPU_BACKEND == "AMDGPU"
        gpu(FluxAMDGPUAdaptor(), x)
    elseif GPU_BACKEND == "Metal"
        gpu(FluxMetalAdaptor(), x)
    elseif GPU_BACKEND == "CPU"
        cpu(x)
    else
        error("""
        Unsupported GPU backend: $GPU_BACKEND.
        Supported backends are: $GPU_BACKENDS.
        """)
    end
end

# Precision

struct FluxEltypeAdaptor{T} end

Adapt.adapt_storage(::FluxEltypeAdaptor{T}, x::AbstractArray{<:AbstractFloat}) where {T<:AbstractFloat} = 
  convert(AbstractArray{T}, x)
Adapt.adapt_storage(::FluxEltypeAdaptor{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T<:AbstractFloat} = 
  convert(AbstractArray{Complex{T}}, x)

_paramtype(::Type{T}, m) where T = fmap(adapt(FluxEltypeAdaptor{T}()), m)

# fastpath for arrays
_paramtype(::Type{T}, x::AbstractArray{<:AbstractFloat}) where {T<:AbstractFloat} = 
  convert(AbstractArray{T}, x)
_paramtype(::Type{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T<:AbstractFloat} = 
  convert(AbstractArray{Complex{T}}, x)

"""
    f32(m)

Converts the `eltype` of model's *floating point* parameters to `Float32` (which is Flux's default).
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

See also [`f64`](@ref) and [`f16`](@ref).
"""
f32(m) = _paramtype(Float32, m)

"""
    f64(m)

Converts the `eltype` of model's *floating point* parameters to `Float64`.
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

See also [`f32`](@ref) and [`f16`](@ref).
"""
f64(m) = _paramtype(Float64, m)

"""
    f16(m)

Converts the `eltype` of model's *floating point* parameters to `Float16`.
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

Support for `Float16` is limited on many CPUs. Julia may
convert to `Float32` for each operation, which is slow.

See also [`f32`](@ref) and [`f64`](@ref).

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
f16(m) = _paramtype(Float16, m)

# Functors for certain Julia data structures -- PIRACY, should move to Functors.jl
@functor Cholesky
trainable(c::Cholesky) = ()

# CUDA extension. ########

Base.@kwdef struct FluxCUDAAdaptor
    id::Union{Nothing, Int} = nothing
end

const CUDA_LOADED = Ref{Bool}(false)

function gpu(to::FluxCUDAAdaptor, x)
    if CUDA_LOADED[]
        return _cuda(to.id, x)
    else
        @info """
        The CUDA functionality is being called but
        `CUDA.jl` must be loaded to access it.
        Add `using CUDA` or `import CUDA` to your code.  Alternatively, configure a different GPU backend by calling `Flux.gpu_backend!`.
        """ maxlog=1
        return x
    end
end

function _cuda end

# AMDGPU extension. ########

Base.@kwdef struct FluxAMDGPUAdaptor
    id::Union{Nothing, Int} = nothing
end

const AMDGPU_LOADED = Ref{Bool}(false)

function gpu(to::FluxAMDGPUAdaptor, x)
    if AMDGPU_LOADED[]
        return _amd(to.id, x)
    else
        @info """
        The AMDGPU functionality is being called but
        `AMDGPU.jl` must be loaded to access it.
        Add `using AMDGPU` or `import AMDGPU` to your code.
        """ maxlog=1
        return x
    end
end

function _amd end

# Metal extension. ######

struct FluxMetalAdaptor end

const METAL_LOADED = Ref{Bool}(false)

function gpu(::FluxMetalAdaptor, x)
    if METAL_LOADED[]
        return _metal(x)
    else
        @info """
        The Metal functionality is being called but
        `Metal.jl` must be loaded to access it.
        """ maxlog=1
        return x
    end
end

function _metal end

################################

"""
    gpu(data::DataLoader)
    cpu(data::DataLoader)

Transforms a given `DataLoader` to apply `gpu` or `cpu` to each batch of data,
when iterated over. (If no GPU is available, this does nothing.)

# Example

```julia-repl
julia> dl = Flux.DataLoader((x = ones(2,10), y='a':'j'), batchsize=3)
4-element DataLoader(::NamedTuple{(:x, :y), Tuple{Matrix{Float64}, StepRange{Char, Int64}}}, batchsize=3)
  with first element:
  (; x = 2×3 Matrix{Float64}, y = 3-element StepRange{Char, Int64})

julia> first(dl)
(x = [1.0 1.0 1.0; 1.0 1.0 1.0], y = 'a':1:'c')

julia> c_dl = gpu(dl)
4-element DataLoader(::MLUtils.MappedData{:auto, typeof(gpu), NamedTuple{(:x, :y), Tuple{Matrix{Float64}, StepRange{Char, Int64}}}}, batchsize=3)
  with first element:
  (; x = 2×3 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, y = 3-element StepRange{Char, Int64})

julia> first(c_dl).x
2×3 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
 1.0  1.0  1.0
 1.0  1.0  1.0
```

For large datasets, this is preferred over moving all the data to
the GPU before creating the `DataLoader`, like this:

```julia-repl
julia> Flux.DataLoader((x = ones(2,10), y=2:11) |> gpu, batchsize=3)
4-element DataLoader(::NamedTuple{(:x, :y), Tuple{CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, UnitRange{Int64}}}, batchsize=3)
  with first element:
  (; x = 2×3 CUDA.CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}, y = 3-element UnitRange{Int64})
```

!!! warning
    This only works if `gpu` is applied directly to the `DataLoader`.
    While `gpu` acts recursively on Flux models and many basic Julia structs,
    it will not work on (say) a tuple of `DataLoader`s.
"""
function gpu(d::MLUtils.DataLoader)
  MLUtils.DataLoader(MLUtils.mapobs(gpu, d.data),
    d.batchsize,
    d.buffer,
    d.partial,
    d.shuffle,
    d.parallel,
    d.collate,
    d.rng,
  )
end

function cpu(d::MLUtils.DataLoader)
  MLUtils.DataLoader(MLUtils.mapobs(cpu, d.data),
    d.batchsize,
    d.buffer,
    d.partial,
    d.shuffle,
    d.parallel,
    d.collate,
    d.rng,
  )
end
