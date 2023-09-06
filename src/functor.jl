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
    mode === :auto || throw(ArgumentError("testmode! accepts only the symbol :auto, got :$mode"))
    nothing
  elseif mode isa Union{Bool,Nothing}
    mode
  else
    throw(ArgumentError("testmode! does not accept $(repr(mode)) as the 2nd argument"))
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

_isbitsarray(::AbstractArray{<:Number}) = true
_isbitsarray(::AbstractArray{T}) where T = isbitstype(T)
_isbitsarray(x) = false

_isleaf(::AbstractRNG) = true
_isleaf(x) = _isbitsarray(x) || Functors.isleaf(x)

# the order below is important
const GPU_BACKENDS = ["CUDA", "AMDGPU", "Metal", "CPU"]
const GPU_BACKEND_ORDER = Dict(collect(zip(GPU_BACKENDS, 1:length(GPU_BACKENDS))))
const GPU_BACKEND = @load_preference("gpu_backend", "CUDA")

function gpu_backend!(backend::String)
    if backend == GPU_BACKEND
        @info """
        GPU backend is already set to: $backend.
        No need to do anything else.
        """
        return
    end

    backend in GPU_BACKENDS || throw(ArgumentError("""
    Unsupported GPU backend: $backend.
    Supported backends are: $GPU_BACKENDS.
    """))

    @set_preferences!("gpu_backend" => backend)
    @info """
    New GPU backend set: $backend.
    Restart your Julia session for this change to take effect!
    """
end

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
        @warning "\"AMD\" backend is deprecated. Please use \"AMDGPU\" instead."
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
Recurses into structs marked with [`@functor`](@ref).

See also [`f64`](@ref) and [`f16`](@ref).
"""
f32(m) = _paramtype(Float32, m)

"""
    f64(m)

Converts the `eltype` of model's *floating point* parameters to `Float64`.
Recurses into structs marked with [`@functor`](@ref).

See also [`f32`](@ref) and [`f16`](@ref).
"""
f64(m) = _paramtype(Float64, m)

"""
    f16(m)

Converts the `eltype` of model's *floating point* parameters to `Float16`.
Recurses into structs marked with [`@functor`](@ref).

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

# Functors for certain Julia data structures
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
        Add `using CUDA` or `import CUDA` to your code.
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

Transforms a given `DataLoader` to apply `gpu` to each batch of data,
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

# Defining device interfaces.
"""
    Flux.AbstractDevice <: Function

An abstract type representing `device` objects for different GPU backends. The currently supported backends are `"CUDA"`, `"AMDGPU"`, `"Metal"` and `"CPU"`; the `"CPU"` backend is the fallback case when no GPU is available. GPU extensions of Flux define subtypes of this type.

"""
abstract type AbstractDevice <: Function end

function (device::AbstractDevice)(d::MLUtils.DataLoader)
    MLUtils.DataLoader(MLUtils.mapobs(device, d.data),
        d.batchsize,
        d.buffer,
        d.partial,
        d.shuffle,
        d.parallel,
        d.collate,
        d.rng,
    )
end

function _get_device_name(::T)::String where {T <: AbstractDevice} end

## check device availability; more definitions in corresponding extensions
_isavailable(::Nothing) = false
_isfunctional(::Nothing) = false

_isavailable(::AbstractDevice) = false
_isfunctional(::AbstractDevice) = false

"""
    Flux.FluxCPUDevice <: Flux.AbstractDevice

A type representing `device` objects for the `"CPU"` backend for Flux. This is the fallback case when no GPU is available to Flux.
"""
Base.@kwdef struct FluxCPUDevice <: AbstractDevice end

(::FluxCPUDevice)(x) = cpu(x)
_isavailable(::FluxCPUDevice) = true
_isfunctional(::FluxCPUDevice) = true
_get_device_name(::FluxCPUDevice) = "CPU"

"""
    FluxCUDADevice <: AbstractDevice

A type representing `device` objects for the `"CUDA"` backend for Flux.
"""
Base.@kwdef struct FluxCUDADevice <: AbstractDevice
    deviceID
end

"""
    FluxAMDGPUDevice <: AbstractDevice

A type representing `device` objects for the `"AMDGPU"` backend for Flux.
"""
Base.@kwdef struct FluxAMDGPUDevice <: AbstractDevice
    deviceID
end

"""
    FluxMetalDevice <: AbstractDevice

A type representing `device` objects for the `"Metal"` backend for Flux.
"""
Base.@kwdef struct FluxMetalDevice <: AbstractDevice
    deviceID
end

## device list. order is important
const DEVICES = Ref{Vector{Union{Nothing, AbstractDevice}}}(Vector{Union{Nothing, AbstractDevice}}(nothing, length(GPU_BACKENDS)))
DEVICES[][GPU_BACKEND_ORDER["CPU"]] = FluxCPUDevice()

## get device

"""
    Flux.supported_devices()

Get all supported backends for Flux, in order of preference.

# Example

```jldoctest
julia> using Flux;

julia> Flux.supported_devices()
("CUDA", "AMDGPU", "Metal", "CPU")
```
"""
supported_devices() = GPU_BACKENDS

"""
    Flux.get_device(; verbose=false)::Flux.AbstractDevice

Returns a `device` object for the most appropriate backend for the current Julia session. 

First, the function checks whether a backend preference has been set via the [`Flux.gpu_backend!`](@ref) function. If so, an attempt is made to load this backend. If the corresponding trigger package has been loaded and the backend is functional, a `device` corresponding to the given backend is loaded. Otherwise, the backend is chosen automatically. To update the backend preference, use [`Flux.gpu_backend!`](@ref).

If there is no preference, then for each of the `"CUDA"`, `"AMDGPU"`, `"Metal"` and `"CPU"` backends in the given order, this function checks whether the given backend has been loaded via the corresponding trigger package, and whether the backend is functional. If so, the `device` corresponding to the backend is returned. If no GPU backend is available, a `Flux.FluxCPUDevice` is returned.

If `verbose` is set to `true`, then the function prints informative log messages.

# Examples
For the example given below, the backend preference was set to `"AMDGPU"` via the [`gpu_backend!`](@ref) function.

```julia-repl
julia> using Flux;

julia> model = Dense(2 => 3)
Dense(2 => 3)       # 9 parameters

julia> device = Flux.get_device(; verbose=true)       # this will just load the CPU device
[ Info: Using backend set in preferences: AMDGPU.
┌ Warning: Trying to use backend: AMDGPU but it's trigger package is not loaded.
│ Please load the package and call this function again to respect the preferences backend.
└ @ Flux ~/fluxml/Flux.jl/src/functor.jl:638
[ Info: Using backend: CPU.
(::Flux.FluxCPUDevice) (generic function with 1 method)

julia> model = model |> device
Dense(2 => 3)       # 9 parameters

julia> model.weight
3×2 Matrix{Float32}:
 -0.304362  -0.700477
 -0.861201   0.67825
 -0.176017   0.234188
```

Here is the same example, but using `"CUDA"`:

```julia-repl
julia> using Flux, CUDA;

julia> model = Dense(2 => 3)
Dense(2 => 3)       # 9 parameters

julia> device = Flux.get_device(; verbose=true)
[ Info: Using backend set in preferences: AMDGPU.
┌ Warning: Trying to use backend: AMDGPU but it's trigger package is not loaded.
│ Please load the package and call this function again to respect the preferences backend.
└ @ Flux ~/fluxml/Flux.jl/src/functor.jl:637
[ Info: Using backend: CUDA.
(::Flux.FluxCUDADevice) (generic function with 1 method)

julia> model = model |> device
Dense(2 => 3)       # 9 parameters

julia> model.weight
3×2 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
  0.820013   0.527131
 -0.915589   0.549048
  0.290744  -0.0592499
```
"""
function get_device(; verbose=false)::AbstractDevice
    backend = @load_preference("gpu_backend", nothing) 

    if backend !== nothing
        allowed_backends = supported_devices()
        idx = findfirst(isequal(backend), allowed_backends)
        if backend ∉  allowed_backends
            @warn """
                `gpu_backend` preference is set to $backend, which is not allowed.
                Defaulting to automatic device selection.
            """ maxlog=1
        else
            verbose && @info "Using backend set in preferences: $backend."
            device = DEVICES[][idx] 

            if !_isavailable(device)
                @warn """
                Trying to use backend: $backend but it's trigger package is not loaded.
                Please load the package and call this function again to respect the preferences backend.
                """
            else 
                if _isfunctional(device)
                    return device
                else
                    @warn "Backend: $backend from the set preferences is not functional. Defaulting to automatic device selection."
                end
            end
        end
    end

    for backend in GPU_BACKENDS 
        device = DEVICES[][GPU_BACKEND_ORDER[backend]]
        if _isavailable(device)
            if _isfunctional(device)
                verbose && @info "Using backend: $backend."
                return device
            end
        end
    end
end

"""
    Flux.get_device(backend::String, idx::Int = 0)::Flux.AbstractDevice

Get a device object for a backend specified by the string `backend` and `idx`. The currently supported values
of `backend` are `"CUDA"`, `"AMDGPU"` and `"CPU"`. `idx` must be an integer value between `0` and the number of available devices.

# Examples

```julia-repl
julia> using Flux, CUDA;

julia> CUDA.devices()
CUDA.DeviceIterator() for 3 devices:
0. GeForce RTX 2080 Ti
1. GeForce RTX 2080 Ti
2. TITAN X (Pascal)

julia> device0 = Flux.get_device("CUDA", 0)
(::Flux.FluxCUDADevice) (generic function with 1 method)

julia> device0.deviceID
CuDevice(0): GeForce RTX 2080 Ti

julia> device1 = Flux.get_device("CUDA", 1)
(::Flux.FluxCUDADevice) (generic function with 1 method)

julia> device1.deviceID
CuDevice(1): GeForce RTX 2080 Ti

julia> cpu_device = Flux.get_device("CPU")
(::Flux.FluxCPUDevice) (generic function with 1 method)

```
"""
function get_device(backend::String, idx::Int = 0)
    if backend == "CPU"
        return FluxCPUDevice()
    else
        return get_device(Val(Symbol(backend)), idx)
    end
end

# Fallback
function get_device(::Val{D}, idx) where D
    error("Unsupported backend: $(D). Try importing the corresponding package.")
end
