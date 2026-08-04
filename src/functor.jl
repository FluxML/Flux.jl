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

# Layers that override the mixed-precision conversions `f16mix`/`bf16mix` to keep (some
# of) their arrays in `Float32`. Extended for normalization layers in `layers/normalise.jl`.
_keep_f32_under_halfprec(::Any) = false

_paramtype(::Type{T}, m) where T = fmap(adapt(FluxEltypeAdaptor{T}()), m)

# Mixed-precision conversion (`f16mix`/`bf16mix`): stop the walk at layers that manage
# their own precision and hand them to `f32` (so their statistics/affine parameters stay
# in `Float32`); convert every other leaf array to `T`.
function _paramtype_mixed(::Type{T}, m) where T
  fmap(m; exclude = x -> _keep_f32_under_halfprec(x) || Functors.isleaf(x)) do x
    _keep_f32_under_halfprec(x) ? f32(x) : adapt(FluxEltypeAdaptor{T}(), x)
  end
end

# fastpath for arrays
_paramtype(::Type{T}, x::AbstractArray{<:AbstractFloat}) where {T<:AbstractFloat} =
  convert(AbstractArray{T}, x)
_paramtype(::Type{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T<:AbstractFloat} =
  convert(AbstractArray{Complex{T}}, x)

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
_paramtype(::Type{BFloat16}, x::AbstractArray{<:AbstractFloat}) = _to_bf16(x)
_paramtype(::Type{BFloat16}, x::AbstractArray{<:Complex{<:AbstractFloat}}) = _to_bf16(x)

"""
    f32(m)

Converts the `eltype` of model's *floating point* parameters to `Float32` (which is Flux's default).
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

See also [`f64`](@ref), [`f16`](@ref) and [`bf16`](@ref).
"""
f32(m) = _paramtype(Float32, m)

"""
    f64(m)

Converts the `eltype` of model's *floating point* parameters to `Float64`.
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

See also [`f32`](@ref), [`f16`](@ref) and [`bf16`](@ref).
"""
f64(m) = _paramtype(Float64, m)

"""
    f16(m)

Converts the `eltype` of model's *floating point* parameters to `Float16`,
like PyTorch's `model.half()`. All parameters are converted, including the
statistics and affine parameters of the normalization layers; note that the GPU
normalization kernels (cuDNN) require `Float32` statistics/affine parameters for
half-precision inputs, so use [`f16mix`](@ref) for models containing `BatchNorm`,
`InstanceNorm` or `GroupNorm`.
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

Support for `Float16` is limited on many CPUs. Julia may
convert to `Float32` for each operation, which is slow.

For mixed-precision *training* with `Float32` master weights, see [`autocast`](@ref).

See also [`f16mix`](@ref), [`f32`](@ref), [`f64`](@ref) and [`bf16`](@ref).

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

"""
    bf16(m)

Converts the `eltype` of model's *floating point* parameters to `BFloat16`
(from [BFloat16s.jl](https://github.com/JuliaMath/BFloat16s.jl)).
Recurses into structs marked with [`@layer`](@ref Flux.@layer).

`BFloat16` has the same exponent range as `Float32` but a reduced mantissa, trading
precision for range. It is often a better choice than [`f16`](@ref) for training, as it
is less prone to overflow/underflow, and is well supported on modern GPUs.

Support for `BFloat16` is limited on many CPUs, where Julia may convert to `Float32`
for each operation.

All parameters are converted, including the statistics and affine parameters of the
normalization layers; note that the GPU normalization kernels (cuDNN) require `Float32`
statistics/affine parameters for half-precision inputs, so use [`bf16mix`](@ref) for
models containing `BatchNorm`, `InstanceNorm` or `GroupNorm`.

For mixed-precision *training* with `Float32` master weights, see [`autocast`](@ref).

See also [`bf16mix`](@ref), [`f16`](@ref), [`f32`](@ref) and [`f64`](@ref).

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
bf16(m) = _paramtype(BFloat16, m)

"""
    f16mix(m)

Converts the model to *mixed* `Float16` precision: like [`f16`](@ref), but the
statistics and affine parameters of `BatchNorm`, `InstanceNorm` and `GroupNorm` are
kept in `Float32` while the data flowing through them stays in `Float16`. This
matches the functional normalization operators in NNlib (and cuDNN), which require
`Float32` parameters for half-precision feature maps — so unlike a full `f16` cast,
a mixed-precision model works on the GPU. `LayerNorm` contains no such parameters
and is converted fully.

For mixed-precision *training* with `Float32` master weights, see [`autocast`](@ref).
To keep the optimiser state in `Float32` when training a converted model, see
[`Optimisers.MixedPrecision`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.MixedPrecision).

See also [`bf16mix`](@ref), [`f16`](@ref), [`f32`](@ref) and [`f64`](@ref).
"""
f16mix(m) = _paramtype_mixed(Float16, m)

"""
    bf16mix(m)

Converts the model to *mixed* `BFloat16` precision: like [`bf16`](@ref), but the
statistics and affine parameters of `BatchNorm`, `InstanceNorm` and `GroupNorm` are
kept in `Float32` while the data flowing through them stays in `BFloat16`. This
matches the functional normalization operators in NNlib (and cuDNN), which require
`Float32` parameters for half-precision feature maps — so unlike a full `bf16` cast,
a mixed-precision model works on the GPU. `LayerNorm` contains no such parameters
and is converted fully.

For mixed-precision *training* with `Float32` master weights, see [`autocast`](@ref).
To keep the optimiser state in `Float32` when training a converted model, see
[`Optimisers.MixedPrecision`](https://fluxml.ai/Optimisers.jl/dev/api/#Optimisers.MixedPrecision).

See also [`f16mix`](@ref), [`bf16`](@ref), [`f32`](@ref) and [`f64`](@ref).
"""
bf16mix(m) = _paramtype_mixed(BFloat16, m)
