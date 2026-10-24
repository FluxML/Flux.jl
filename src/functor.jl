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

"""
    trainmode!(model) -> model

Set a layer, or all layers in a model, to training mode.
Opposite to [`testmode!`](@ref), see further details there.
"""
trainmode!(m) = testmode!(m, false)
trainmode!(m, mode::Symbol) = testmode!(m, mode)
trainmode!(m, ::Nothing) = testmode!(m, nothing)  # why do we have so much API?



# CPU/GPU movement conveniences

"""
    cpu(m)

Copies `m` onto the CPU, the opposite of [`gpu`](@ref).
Recurses into structs (thanks to Functors.jl).

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
cpu(x) = cpu_device()(x)

"""
    gpu(m)

Copies `m` to the current GPU device (using current GPU backend), if one is available.
If no GPU is available, it does nothing (but prints a warning the first time).
It recurses into structs according to Functors.jl.

Use [`cpu`](@ref) to copy back to ordinary `Array`s.
See also [`f32`](@ref) and [`f16`](@ref) to change element type only.

This function is just defined for convenience around [`gpu_device`](@ref), 
and is equivalent to `gpu_device()(m)`.
You may consider defining `device = gpu_device()` once and then using `device(m)` to move data.

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
gpu(x) = gpu_device()(x)

# Precision

struct FluxEltypeAdaptor{T} end

Adapt.adapt_storage(::FluxEltypeAdaptor{T}, x::AbstractArray{<:AbstractFloat}) where {T<:AbstractFloat} = 
  convert(AbstractArray{T}, x)
Adapt.adapt_storage(::FluxEltypeAdaptor{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T<:AbstractFloat} = 
  convert(AbstractArray{Complex{T}}, x)

# One call of `f16`, `bf16`, `f32` or `f64` creates a `CastContext` and passes it to every
# 2-argument method it reaches. It holds a cache per target eltype, so that an object
# reached twice (e.g. tied weights) is converted once, even across a layer's own method.
struct CastContext
  f16::IdDict{Any,Any}
  bf16::IdDict{Any,Any}
  f32::IdDict{Any,Any}
  f64::IdDict{Any,Any}
end
CastContext() = CastContext(IdDict{Any,Any}(), IdDict{Any,Any}(), IdDict{Any,Any}(), IdDict{Any,Any}())
ChainRulesCore.@non_differentiable CastContext()

_cache(cx::CastContext, ::Type{Float16}) = cx.f16
_cache(cx::CastContext, ::Type{BFloat16}) = cx.bf16
_cache(cx::CastContext, ::Type{Float32}) = cx.f32
_cache(cx::CastContext, ::Type{Float64}) = cx.f64

_castfn(::Type{Float16}) = f16
_castfn(::Type{BFloat16}) = bf16
_castfn(::Type{Float32}) = f32
_castfn(::Type{Float64}) = f64

# Like Functors' cache: identity matters for mutable objects and arrays.
_usecache(x) = ismutable(x) || x isa AbstractArray

# The default conversion of a node: convert a leaf, or call the 2-argument method of
# `f16` etc. on each child, so that a layer's own method applies anywhere in a model.
function _paramtype(::Type{T}, x, cx::CastContext) where T
  _usecache(x) || return _convert(T, x, cx)
  cache = _cache(cx, T)
  haskey(cache, x) && return cache[x]::Base.promote_op(_convert, Type{T}, typeof(x), typeof(cx))
  return cache[x] = _convert(T, x, cx)
end

function _convert(::Type{T}, x, cx) where T
  Functors.isleaf(x) && return adapt(FluxEltypeAdaptor{T}(), x)
  f = _castfn(T)
  return Functors.DefaultWalk()(c -> f(c, cx), x)
end

# A conversion keeps the structure of the model, so its pullback maps each cotangent back
# onto the input, projected to the input's eltype. This also keeps Zygote from tracing
# the walk, and gives each use of a tied array its own gradient.
function ChainRulesCore.rrule(::typeof(_paramtype), ::Type{T}, x, cx::CastContext) where T
  _paramtype_back(dy) = (NoTangent(), NoTangent(), _paramtype_pullback(x, dy), NoTangent())
  return _paramtype(T, x, cx), _paramtype_back
end

function _paramtype_pullback(x, dy)
  dy = unthunk(dy)
  dy isa Union{Nothing, AbstractZero} ? dy : _project_back(x, dy)
end
_project_back(x, dy) = dy
_project_back(x::AbstractArray{<:Number}, dy) = ProjectTo(x)(dy)
_project_back(x::AbstractArray, dy::AbstractArray) = map(_paramtype_pullback, x, dy)
_project_back(x::AbstractArray{<:Number}, dy::AbstractArray) = ProjectTo(x)(dy)
_project_back(x::Tuple, dy::Tuple) = map(_paramtype_pullback, x, dy)
_project_back(x, dy::Tangent) = _project_back(x, ChainRulesCore.backing(dy))
_project_back(x, dy::NamedTuple{K}) where K =
  NamedTuple{K}(map(k -> _paramtype_pullback(getfield(x, k), getfield(dy, k)), K))

# BFloat16 needs a hand-rolled conversion. On x86-64 with LLVM 18 (Julia 1.12.x) the
# native `Float32`->`BFloat16` round (`Base.fptrunc`) auto-vectorizes into an
# `X86ISD::VFPROUND` that LLVM cannot select, so `convert(AbstractArray{BFloat16}, ::Array{Float32})`
# crashes or deadlocks (JuliaMath/BFloat16s.jl#107). We round to nearest even in the
# integer domain instead (bit-identical to the native result, never emits `fptrunc`).
@inline function _to_bf16(x::Real)
  f = Float32(x)
  u = reinterpret(UInt32, f)
  bits = ((u + 0x00007fff + ((u >> 16) & 0x00000001)) >> 16) % UInt16
  return reinterpret(BFloat16, ifelse(isnan(f), 0x7fc0 % UInt16, bits))
end
_to_bf16(x::AbstractArray{<:Real}) = _to_bf16.(x)
_to_bf16(x::Adjoint{<:Real}) = adjoint(_to_bf16(parent(x)))       # keep the wrapper, like `convert`
_to_bf16(x::Transpose{<:Real}) = transpose(_to_bf16(parent(x)))
_to_bf16(x::AbstractArray{<:Complex}) = complex.(_to_bf16(real(x)), _to_bf16(imag(x)))

# `_to_bf16` rounds via non-differentiable integer ops; give it a straight-through
# pullback so `bf16` stays differentiable (matching `f16`/`f32`/`f64`).
function ChainRulesCore.rrule(::typeof(_to_bf16), x::AbstractArray)
  proj = ChainRulesCore.ProjectTo(x)
  _to_bf16_back(Δ) = (ChainRulesCore.NoTangent(), proj(Δ))
  return _to_bf16(x), _to_bf16_back
end

Adapt.adapt_storage(::FluxEltypeAdaptor{BFloat16}, x::AbstractArray{<:AbstractFloat}) = _to_bf16(x)
Adapt.adapt_storage(::FluxEltypeAdaptor{BFloat16}, x::AbstractArray{<:Complex{<:AbstractFloat}}) = _to_bf16(x)

"""
    f32(m)
    f32(m, cx)

Converts the `eltype` of model's *floating point* parameters to `Float32` (which is Flux's default).
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

The recursion calls the 2-argument method `f32(child, cx)` on each part of the model, so a
layer can change how it is converted by adding a method of it, see
[Customising the precision of a layer](@ref man-custom-precision).

See also [`f64`](@ref), [`f16`](@ref) and [`bf16`](@ref).
"""
f32(m) = f32(m, CastContext())
f32(m, cx) = _paramtype(Float32, m, cx)

"""
    f64(m)
    f64(m, cx)

Converts the `eltype` of model's *floating point* parameters to `Float64`.
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

The recursion calls the 2-argument method `f64(child, cx)` on each part of the model, so a
layer can change how it is converted by adding a method of it, see
[Customising the precision of a layer](@ref man-custom-precision).

See also [`f32`](@ref), [`f16`](@ref) and [`bf16`](@ref).
"""
f64(m) = f64(m, CastContext())
f64(m, cx) = _paramtype(Float64, m, cx)

"""
    f16(m)
    f16(m, cx)

Converts the `eltype` of model's *floating point* parameters to `Float16`.
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

Support for `Float16` is limited on many CPUs. Julia may
convert to `Float32` for each operation, which is slow.

The normalization layers `BatchNorm`, `InstanceNorm`, `GroupNorm` and `LayerNorm` are
converted in *mixed precision*: their statistics and affine parameters are kept in
`Float32` while the data flowing through stays in `Float16`. This matches the functional
operators in NNlib (and cuDNN), which require `Float32` parameters for half-precision
feature maps.

The recursion calls the 2-argument method `f16(child, cx)` on each part of the model, so a
layer can change how it is converted by adding a method of it. For example, this keeps a
custom normalization layer in `Float32`, like the built-in ones:

```julia
Flux.f16(m::MyNorm, cx) = f32(m, cx)
```

See [Customising the precision of a layer](@ref man-custom-precision) for details.

See also [`f32`](@ref), [`f64`](@ref) and [`bf16`](@ref).

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
f16(m) = f16(m, CastContext())
f16(m, cx) = _paramtype(Float16, m, cx)

"""
    bf16(m)
    bf16(m, cx)

Converts the `eltype` of model's *floating point* parameters to `BFloat16`
(from [BFloat16s.jl](https://github.com/JuliaMath/BFloat16s.jl)).
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

`BFloat16` has the same exponent range as `Float32` but a reduced mantissa, trading
precision for range. It is often a better choice than [`f16`](@ref) for training, as it
is less prone to overflow/underflow, and is well supported on modern GPUs.

Support for `BFloat16` is limited on many CPUs, where Julia may convert to `Float32`
for each operation.

The normalization layers `BatchNorm`, `InstanceNorm`, `GroupNorm` and `LayerNorm` are
converted in *mixed precision*: their statistics and affine parameters are kept in
`Float32` while the data flowing through stays in `BFloat16`. This matches the functional
operators in NNlib (and cuDNN), which require `Float32` parameters for half-precision
feature maps.

The recursion calls the 2-argument method `bf16(child, cx)` on each part of the model, so a
layer can change how it is converted by adding a method of it. For example, this keeps a
custom normalization layer in `Float32`, like the built-in ones:

```julia
Flux.bf16(m::MyNorm, cx) = f32(m, cx)
```

See [Customising the precision of a layer](@ref man-custom-precision) for details.

See also [`f16`](@ref), [`f32`](@ref) and [`f64`](@ref).

# Example
```jldoctest
julia> m = Chain(Dense(784, 2048, relu), Dense(2048, 10))  # all Float32
Chain(
  Dense(784 => 2048, relu),             # 1_607_680 parameters
  Dense(2048 => 10),                    # 20_490 parameters
)                   # Total: 4 arrays, 1_628_170 parameters, 6.211 MiB.

julia> m |> bf16  # takes half the memory
Chain(
  Dense(784 => 2048, relu),             # 1_607_680 parameters
  Dense(2048 => 10),                    # 20_490 parameters
)                   # Total: 4 arrays, 1_628_170 parameters, 3.106 MiB.
```
"""
bf16(m) = bf16(m, CastContext())
bf16(m, cx) = _paramtype(BFloat16, m, cx)
